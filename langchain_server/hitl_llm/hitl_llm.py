"""
HITL (Human-in-the-Loop) sample using an LLM.
Single-file Flask app that demonstrates:
 1) LLM generates a draft for a user prompt
 2) LLM self-evaluates confidence (or we simulate it)
 3) Low-confidence results are queued for human review
 4) A simple web UI allows a human reviewer to Approve / Edit / Reject
 5) Approved items are published (stored) and visible

Requirements:
  - Python 3.9+
  - pip install flask openai tinydb

Usage:
  1) Set OPENAI_API_KEY environment variable (or adapt code to your LLM client)
  2) python LLM_HITL_sample.py
  3) Open http://localhost:5000

Notes:
  - This is a minimal educational sample. In production:
    * Use authentication for reviewers
    * Persist to a proper database
    * Add audit/logging and RBAC
    * Implement queueing, retries, notification (Slack/email)

"""

from flask import Flask, request, redirect, url_for, render_template_string
import os
import uuid
import random
from tinydb import TinyDB, Query

# --- Simple "DB" ---
db = TinyDB('hitl_db.json')
pending_table = db.table('pending')
published_table = db.table('published')

# --- Flask app ---
app = Flask(__name__)

# --- Replace this with a real LLM call (Ollama local LLM) ---
# Requires: pip install ollama
from ollama import Client
import numpy as np
ollama_client = Client()

def get_embedding(text: str) -> list:
    """テキストのembeddingベクトルを取得"""
    try:
        res = ollama_client.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )
        return res.get("embedding", [])
    except Exception as e:
        print(f"Embedding error: {e}")
        return []

def cosine_similarity(vec1: list, vec2: list) -> float:
    """2つのベクトル間のコサイン類似度を計算"""
    if not vec1 or not vec2:
        return 0.0
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def find_similar_approved_answers(prompt: str, limit: int = 3, threshold: float = 0.5) -> list:
    """公開済みの回答から意味的に類似した質問を検索"""
    published_items = published_table.all()
    if not published_items:
        return []
    
    # 入力プロンプトのembeddingを取得
    prompt_embedding = get_embedding(prompt)
    if not prompt_embedding:
        print("Failed to get prompt embedding, falling back to no context")
        return []
    
    # 各公開済み質問との意味的類似度を計算
    similar_items = []
    
    for item in published_items:
        item_prompt = item.get('prompt', '')
        if not item_prompt:
            continue
        
        # 公開済み質問のembeddingを取得
        item_embedding = get_embedding(item_prompt)
        if not item_embedding:
            continue
        
        # コサイン類似度を計算
        similarity_score = cosine_similarity(prompt_embedding, item_embedding)
        
        # 閾値以上の類似度の場合のみ追加
        if similarity_score >= threshold:
            similar_items.append({
                'item': item,
                'score': similarity_score
            })
    
    # 類似度スコア順にソート
    similar_items.sort(key=lambda x: x['score'], reverse=True)
    
    # デバッグ情報
    if similar_items:
        print(f"Found {len(similar_items)} similar items (threshold={threshold}):")
        for i, si in enumerate(similar_items[:limit], 1):
            print(f"  {i}. Score: {si['score']:.3f} - {si['item'].get('prompt', '')[:50]}...")
    
    return [x['item'] for x in similar_items[:limit]]

def build_context_from_approved(similar_items: list) -> str:
    """承認済み回答から参考コンテキストを構築"""
    if not similar_items:
        return ""
    
    context = "\n\n以下は過去に承認された類似の質問と回答です：\n\n"
    for i, item in enumerate(similar_items, 1):
        context += f"【参考例 {i}】\n"
        context += f"質問: {item.get('prompt', '')}\n"
        context += f"承認済み回答: {item.get('output', '')}\n\n"
    
    return context

def llm_generate(prompt: str) -> str:
    """Call local Ollama model with approved context."""
    # 承認済みの類似回答を検索
    similar_items = find_similar_approved_answers(prompt)
    
    # コンテキストを構築
    if similar_items:
        context = build_context_from_approved(similar_items)
        enhanced_prompt = f"""{context}
上記の承認済み回答を参考にして、以下の質問に答えてください。
過去の承認された回答のスタイルと品質を維持しながら、新しい質問に適切に回答してください。

質問: {prompt}

回答:"""
        print(f"Found {len(similar_items)} similar approved answers, using as context")
    else:
        enhanced_prompt = prompt
        print("No similar approved answers found, generating from scratch")

    res = ollama_client.generate(model="qwen3:8b", prompt=enhanced_prompt)
    # res = ollama_client.generate(model="llama3:latest", prompt=enhanced_prompt)
    return res.get("response", "")

def llm_self_eval(text: str) -> float:
    """Ask the local LLM for a confidence rating 0.0〜1.0."""
    eval_prompt = f"あなたは自己評価エンジンです。以下の回答の信頼度を0.0〜1.0で1つの数値のみ返してください。回答:{text}"
    # res = ollama_client.generate(model="llama3:latest", prompt=eval_prompt)
    res = ollama_client.generate(model="qwen3:14b", prompt=eval_prompt)
    print("Self-eval response:", res)
    try:
        return float(res.get("response", "0.5").strip())
    except:
        return 0.5

# keep compatibility names
mock_llm_generate = llm_generate
mock_llm_self_eval = llm_self_eval


# --- HTML templates (embedded for single-file simplicity) ---
INDEX_HTML = """
<h2>LLM HITL Demo</h2>
<form action="/submit" method="post">
  <label>Prompt (question / task):</label><br>
  <textarea name="prompt" rows="4" cols="60"></textarea><br>
  <button type="submit">Generate</button>
</form>
<hr>
<p><a href="/review">Pending for review</a> | <a href="/published">Published</a></p>
"""

REVIEW_LIST_HTML = """
<h2>Pending items for human review</h2>
{% if items %}
  <ul>
  {% for it in items %}
    <li>
      <strong>{{it.prompt}}</strong><br>
      <pre>{{it.output}}</pre>
      Confidence: {{'%.2f'|format(it.confidence)}}<br>
      <form style="display:inline" action="/approve/{{it.id}}" method="post"><button>Approve</button></form>
      <form style="display:inline" action="/reject/{{it.id}}" method="post"><button>Reject</button></form>
      <form style="display:inline" action="/edit/{{it.id}}" method="get"><button>Edit</button></form>
    </li>
  {% endfor %}
  </ul>
{% else %}
  <p>No pending items.</p>
{% endif %}
<p><a href="/">Back</a></p>
"""

EDIT_HTML = """
<h2>Edit item</h2>
<form action="/edit/{{it.id}}" method="post">
  <input type="hidden" name="id" value="{{it.id}}">
  <label>Prompt:</label><br>
  <div>{{it.prompt}}</div><br>
  <label>Output (edit if needed):</label><br>
  <textarea name="output" rows="8" cols="70">{{it.output}}</textarea><br>
  <button type="submit">Save & Approve</button>
</form>
<p><a href="/review">Back</a></p>
"""

PUBLISHED_HTML = """
<h2>Published items</h2>
{% if items %}
  <ul>
  {% for it in items %}
    <li><strong>{{it.prompt}}</strong><pre>{{it.output}}</pre><small>by human: {{it.human_id}}</small></li>
  {% endfor %}
  </ul>
{% else %}
  <p>No published items.</p>
{% endif %}
<p><a href="/">Back</a></p>
"""


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/submit', methods=['POST'])
def submit():
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return redirect(url_for('index'))

    # 0) Check for similar approved answers
    similar_items = find_similar_approved_answers(prompt)
    context_info = ""
    if similar_items:
        context_info = f"<p>💡 {len(similar_items)}件の類似した承認済み回答を参考にしました</p>"
    
    # 1) Generate candidate with LLM (with approved context)
    print(f"Generating for prompt: {prompt}")
    output = mock_llm_generate(prompt)

    # 2) Get self-evaluated confidence
    confidence = mock_llm_self_eval(output)

    # 3) Threshold -> if low confidence, queue for human review
    THRESHOLD = 0.85
    if confidence < THRESHOLD:
        item = {
            'id': uuid.uuid4().hex,
            'prompt': prompt,
            'output': output,
            'confidence': confidence,
            'created_at': None,
        }
        pending_table.insert(item)
        return f"{context_info}Result queued for human review (confidence={confidence:.2f}). <a href=\"/review\">Go to review queue</a>"
    else:
        # publish automatically
        # published_table.insert({
        #     'id': uuid.uuid4().hex,
        #     'prompt': prompt,
        #     'output': output,
        #     'confidence': confidence,
        #     'human_id': 'auto',
        # })
        print("output "+ output)
        return f"{context_info}{output}<p>Published automatically (confidence={confidence:.2f}). <a href=\"/published\">View published</a>"


@app.route('/review')
def review():
    items = pending_table.all()
    # convert confidence for template
    for i in items:
        i['confidence'] = i.get('confidence', 0.0)
    return render_template_string(REVIEW_LIST_HTML, items=items)


@app.route('/approve/<item_id>', methods=['POST'])
def approve(item_id):
    Item = Query()
    res = pending_table.get(Item.id == item_id)
    if not res:
        return 'Not found', 404
    # In a real system record which human approved (user auth)
    published_table.insert({
        'id': uuid.uuid4().hex,
        'prompt': res['prompt'],
        'output': res['output'],
        'confidence': res.get('confidence', 0.0),
        'human_id': 'human_approver',
    })
    pending_table.remove(Item.id == item_id)
    return redirect(url_for('review'))


@app.route('/reject/<item_id>', methods=['POST'])
def reject(item_id):
    Item = Query()
    pending_table.remove(Item.id == item_id)
    return redirect(url_for('review'))


@app.route('/edit/<item_id>', methods=['GET', 'POST'])
def edit(item_id):
    Item = Query()
    if request.method == 'GET':
        res = pending_table.get(Item.id == item_id)
        if not res:
            return 'Not found', 404
        return render_template_string(EDIT_HTML, it=res)
    else:
        new_output = request.form.get('output', '')
        res = pending_table.get(Item.id == item_id)
        if not res:
            return 'Not found', 404
        # Save edited and publish
        published_table.insert({
            'id': uuid.uuid4().hex,
            'prompt': res['prompt'],
            'output': new_output,
            'confidence': res.get('confidence', 0.0),
            'human_id': 'human_editor',
        })
        pending_table.remove(Item.id == item_id)
        return redirect(url_for('review'))


@app.route('/published')
def published():
    items = published_table.all()
    return render_template_string(PUBLISHED_HTML, items=items)


if __name__ == '__main__':
    # Simple dev server
    app.run(host='0.0.0.0', port=8000, debug=True)
