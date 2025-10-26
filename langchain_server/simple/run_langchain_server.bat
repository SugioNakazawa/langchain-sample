@echo off
REM LangChain Server 実行スクリプト (Windows用)
REM 使用方法: run_langchain_server.bat

echo === LangChain Server 起動スクリプト ===

REM 設定
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
set VENV_PATH=%PROJECT_ROOT%\venv
set SERVER_FILE=%SCRIPT_DIR%langchain_server.py
set SERVER_HOST=0.0.0.0
set SERVER_PORT=8000

echo プロジェクトディレクトリ: %PROJECT_ROOT%
echo 仮想環境: %VENV_PATH%
echo サーバーファイル: %SERVER_FILE%

REM 仮想環境の確認
if not exist "%VENV_PATH%" (
    echo エラー: 仮想環境が見つかりません: %VENV_PATH%
    echo 以下のコマンドで仮想環境を作成してください:
    echo   cd %PROJECT_ROOT%
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install fastapi uvicorn langchain langchain-openai
    pause
    exit /b 1
)

REM サーバーファイルの確認
if not exist "%SERVER_FILE%" (
    echo エラー: サーバーファイルが見つかりません: %SERVER_FILE%
    pause
    exit /b 1
)

REM 仮想環境のアクティベート
echo 仮想環境をアクティベートしています...
call "%VENV_PATH%\Scripts\activate.bat"

REM Ollamaサーバーの確認
echo Ollamaサーバーの確認中...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: Ollamaサーバーに接続できません (http://localhost:11434^)
    echo Ollamaを起動してください: ollama serve
    echo 続行しますか? (y/N^)
    set /p response=
    if /i not "%response%"=="y" exit /b 1
) else (
    echo Ollamaサーバーが利用可能です
)

REM サーバー起動
echo LangChain Serverを起動しています...
echo URL: http://%SERVER_HOST%:%SERVER_PORT%
echo API: http://%SERVER_HOST%:%SERVER_PORT%/v1/chat/completions
echo Models: http://%SERVER_HOST%:%SERVER_PORT%/v1/models
echo 停止するには Ctrl+C を押してください
echo.

cd /d "%PROJECT_ROOT%"
python "%SERVER_FILE%"

pause