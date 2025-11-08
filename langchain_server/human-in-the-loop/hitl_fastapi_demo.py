"""
HITL (Human-In-The-Loop) サンプル実装 — single-file FastAPI アプリ

ファイル名: hitl_fastapi_demo.py

概要:
 - 簡易的なテキスト分類ワークフローを示すデモ。
 - ユーザーは /submit でテキストを送信 -> モデルが予測 -> 予測結果は DB (SQLite) に保存されステータスは "pending"。
 - 管理者は /review で保留中のアイテムを確認・修正 (ラベル付け) -> 修正はデータベースに保存され、"accepted" または "rejected" に更新される。
 - 再学習ボタンでレビュー済データを使ってモデルを再学習 (scikit-learn を利用)。
 - 実際のプロダクション向けではなく、HITL の概念とパイプラインを示す教育用サンプルです。

依存パッケージ:
 pip install fastapi uvicorn jinja2 scikit-learn python-multipart

実行方法:
 python hitl_fastapi_demo.py
 もしくは uvicorn hitl_fastapi_demo:app --reload

エンドポイント (主なもの):
 - GET  /            : サンプルトップ (送信用フォーム)
 - POST /submit      : テキスト送信 (フォーム)
 - GET  /items       : JSON で全アイテム一覧
 - GET  /review      : 管理者向けレビューUI
 - POST /review/{id} : レビュー結果を保存 (label + action)
 - POST /retrain     : モデル再学習をトリガー

注意:
 - このサンプルはローカル実行を前提としています。
 - 簡素化のため認証は実装していません。管理UI を公開する場合は必ず認証を追加してください。

"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Template
import sqlite3
import threading
import os
import datetime

# optional sklearn imports (used if available for a real retrain)
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    import pickle
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

DB_PATH = "hitl_demo.db"
MODEL_PATH = "hitl_model.pkl"

# --- DB helpers -------------------------------------------------

def init_db():
    need_init = not os.path.exists(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        predicted_label TEXT,
        true_label TEXT,
        status TEXT NOT NULL,
        created_at TEXT,
        reviewed_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def insert_item(text, predicted_label=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO items (text, predicted_label, true_label, status, created_at) VALUES (?,?,?,?,?)",
              (text, predicted_label, None, 'pending', now))
    conn.commit()
    item_id = c.lastrowid
    conn.close()
    return item_id


def update_item_review(item_id, true_label, action):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    status = 'accepted' if action == 'accept' else 'rejected'
    c.execute("UPDATE items SET true_label=?, status=?, reviewed_at=? WHERE id=?",
              (true_label, status, now, item_id))
    conn.commit()
    conn.close()


def get_all_items():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, text, predicted_label, true_label, status, created_at, reviewed_at FROM items ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    keys = ['id','text','predicted_label','true_label','status','created_at','reviewed_at']
    return [dict(zip(keys, r)) for r in rows]


def get_pending_items():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, text, predicted_label FROM items WHERE status='pending' ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return [{'id':r[0],'text':r[1],'predicted_label':r[2]} for r in rows]


def get_reviewed_examples():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT text, true_label FROM items WHERE true_label IS NOT NULL")
    rows = c.fetchall()
    conn.close()
    return rows

# --- Simple model wrapper ---------------------------------------
class SimpleModel:
    def __init__(self):
        self.vectorizer = None
        self.clf = None
        if SKLEARN_AVAILABLE and os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, 'rb') as f:
                    self.vectorizer, self.clf = pickle.load(f)
            except Exception:
                self.vectorizer = None
                self.clf = None

    def predict(self, texts):
        # if sklearn available and model trained -> use it
        if SKLEARN_AVAILABLE and self.vectorizer is not None and self.clf is not None:
            X = self.vectorizer.transform(texts)
            preds = self.clf.predict(X)
            return preds.tolist()
        # otherwise fallback to a naive heuristic
        return [self.heuristic_predict(t) for t in texts]

    def heuristic_predict(self, t):
        # naive rules example: check keywords
        t_low = t.lower()
        if any(w in t_low for w in ['error','fail','problem','issue']):
            return 'negative'
        if any(w in t_low for w in ['love','great','good','nice','ok']):
            return 'positive'
        return 'neutral'

    def retrain_from_db(self):
        if not SKLEARN_AVAILABLE:
            return {'status':'sklearn_unavailable', 'message':'scikit-learn not installed. Install scikit-learn to enable retraining.'}
        examples = get_reviewed_examples()
        if len(examples) < 3:
            return {'status':'not_enough_data', 'message':'Need at least 3 labeled examples to retrain (found {}).'.format(len(examples))}
        texts = [e[0] for e in examples]
        labels = [e[1] for e in examples]
        vec = CountVectorizer(ngram_range=(1,2), min_df=1)
        X = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, labels)
        # save
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump((vec, clf), f)
        self.vectorizer = vec
        self.clf = clf
        return {'status':'ok', 'message':'Retrained on {} examples'.format(len(labels))}

model = SimpleModel()

# --- FastAPI app ------------------------------------------------
app = FastAPI()
init_db()

# Templates (simple, embedded)
INDEX_HTML = Template('''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>HITL Demo - Submit</title>
  </head>
  <body>
    <h1>HITL Demo - Submit Text</h1>
    <form action="/submit" method="post">
      <textarea name="text" rows="6" cols="80" placeholder="Enter text to classify"></textarea><br>
      <button type="submit">Submit</button>
    </form>
    <p>
      <a href="/review">Go to review UI (admin)</a> |
      <a href="/items">JSON: all items</a>
    </p>
  </body>
</html>
''')

REVIEW_HTML = Template('''
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>HITL Demo - Review</title>
  </head>
  <body>
    <h1>Review Queue</h1>
    <h2>Pending</h2>
    {% if pending %}
      <ul>
      {% for p in pending %}
        <li>
          <form action="/review/{{p.id}}" method="post">
            <strong>ID {{p.id}}</strong>: {{p.text}}<br>
            Predicted: <em>{{p.predicted_label}}</em><br>
            True label: <input name="true_label" placeholder="e.g. positive" />
            <button name="action" value="accept" type="submit">Accept</button>
            <button name="action" value="reject" type="submit">Reject</button>
          </form>
        </li>
      {% endfor %}
      </ul>
    {% else %}
      <p>No pending items.</p>
    {% endif %}

    <h2>All items</h2>
    <table border="1" cellpadding="4">
      <tr><th>id</th><th>text</th><th>pred</th><th>true</th><th>status</th></tr>
      {% for it in all_items %}
        <tr>
          <td>{{it.id}}</td>
          <td>{{it.text}}</td>
          <td>{{it.predicted_label}}</td>
          <td>{{it.true_label or ''}}</td>
          <td>{{it.status}}</td>
        </tr>
      {% endfor %}
    </table>

    <h3>Retrain model (uses reviewed examples)</h3>
    <form action="/retrain" method="post">
      <button type="submit">Retrain</button>
    </form>

    <p><a href="/">Back to submit</a></p>
  </body>
</html>
''')

SUCCESS_HTML = Template('''
<!doctype html>
<html>
  <head><meta charset="utf-8"><title>OK</title></head>
  <body>
    <p>{{message}}</p>
    <p><a href="/">Back</a> | <a href="/review">Review</a></p>
  </body>
</html>
''')

@app.get('/', response_class=HTMLResponse)
async def index():
    return INDEX_HTML.render()

@app.post('/submit')
async def submit(text: str = Form(...)):
    # model prediction
    pred = model.predict([text])[0]
    item_id = insert_item(text, predicted_label=pred)
    return HTMLResponse(content=SUCCESS_HTML.render(message=f"Submitted as id={item_id}, predicted={pred}"))

@app.get('/items')
async def items():
    return JSONResponse(get_all_items())

@app.get('/review', response_class=HTMLResponse)
async def review_page():
    pending = get_pending_items()
    all_items = get_all_items()
    return REVIEW_HTML.render(pending=pending, all_items=all_items)

@app.post('/review/{item_id}')
async def review_item(item_id: int, true_label: str = Form(...), action: str = Form(...)):
    update_item_review(item_id, true_label, action)
    return RedirectResponse(url='/review', status_code=303)

@app.post('/retrain')
async def retrain():
    res = model.retrain_from_db()
    # simple response
    if isinstance(res, dict) and res.get('status') == 'ok':
        message = res.get('message')
    else:
        message = res.get('message') if isinstance(res, dict) else str(res)
    return HTMLResponse(content=SUCCESS_HTML.render(message=message))

# optional: background thread to periodically print counts (demo purposes)
def background_monitor():
    import time
    while True:
        all_items = get_all_items()
        pending = [i for i in all_items if i['status']=='pending']
        print(f"[HITL MON] total={len(all_items)} pending={len(pending)}")
        time.sleep(60)

if __name__ == '__main__':
    # start background monitor
    t = threading.Thread(target=background_monitor, daemon=True)
    t.start()
    # run uvicorn
    import uvicorn
    uvicorn.run('hitl_fastapi_demo:app', host='127.0.0.1', port=8000, reload=False)
