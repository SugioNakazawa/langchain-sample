@echo off
REM MCP Blend Server 実行スクリプト (Windows用)
REM 使用方法: run_mcp_blend_server.bat

echo === MCP Blend Server 起動スクリプト ===

REM 設定
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
set VENV_PATH=%PROJECT_ROOT%\venv
set MCP_SERVER_FILE=%SCRIPT_DIR%mcp_blend_server.py
set APP_SERVER_FILE=%SCRIPT_DIR%mcp_app_server.py
set MCP_PORT=9100
set APP_PORT=8000

echo プロジェクトディレクトリ: %PROJECT_ROOT%
echo 仮想環境: %VENV_PATH%
echo MCPサーバーファイル: %MCP_SERVER_FILE%
echo Appサーバーファイル: %APP_SERVER_FILE%

REM 仮想環境の確認
if not exist "%VENV_PATH%" (
    echo エラー: 仮想環境が見つかりません: %VENV_PATH%
    echo 以下のコマンドで仮想環境を作成してください:
    echo   cd %PROJECT_ROOT%
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo   pip install fastapi uvicorn langchain langchain-openai pulp requests
    pause
    exit /b 1
)

REM サーバーファイルの確認
if not exist "%MCP_SERVER_FILE%" (
    echo エラー: MCPサーバーファイルが見つかりません: %MCP_SERVER_FILE%
    pause
    exit /b 1
)

if not exist "%APP_SERVER_FILE%" (
    echo エラー: Appサーバーファイルが見つかりません: %APP_SERVER_FILE%
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

REM 起動方法の選択
echo.
echo 起動方法を選択してください:
echo 1^) MCPサーバーのみ起動 (ポート %MCP_PORT%^)
echo 2^) Appサーバーのみ起動 (ポート %APP_PORT%^)
echo 3^) 両方のサーバーを起動 (MCP: %MCP_PORT%, App: %APP_PORT%^)
echo 4^) テスト実行
set /p choice=選択 (1-4^): 

if "%choice%"=="1" goto start_mcp
if "%choice%"=="2" goto start_app
if "%choice%"=="3" goto start_both
if "%choice%"=="4" goto test_run
echo 無効な選択です
pause
exit /b 1

:start_mcp
echo MCPサーバーを起動しています...
echo URL: http://0.0.0.0:%MCP_PORT%
echo 停止するには Ctrl+C を押してください
cd /d "%PROJECT_ROOT%"
python "%MCP_SERVER_FILE%"
goto end

:start_app
echo Appサーバーを起動しています...
echo URL: http://0.0.0.0:%APP_PORT%
echo API: http://0.0.0.0:%APP_PORT%/v1/chat/completions
echo Models: http://0.0.0.0:%APP_PORT%/v1/models
echo 停止するには Ctrl+C を押してください
cd /d "%PROJECT_ROOT%"
python "%APP_SERVER_FILE%"
goto end

:start_both
echo 両方のサーバーを起動しています...
cd /d "%PROJECT_ROOT%"
echo MCPサーバーをバックグラウンドで起動中...
start /b python "%MCP_SERVER_FILE%" > mcp_server.log 2>&1
echo MCPサーバー起動完了 (ポート: %MCP_PORT%^)
timeout /t 2 /nobreak >nul
echo Appサーバーを起動中...
echo URL: http://0.0.0.0:%APP_PORT%
echo API: http://0.0.0.0:%APP_PORT%/v1/chat/completions
echo Models: http://0.0.0.0:%APP_PORT%/v1/models
echo 停止するには Ctrl+C を押してください
python "%APP_SERVER_FILE%"
goto end

:test_run
echo テスト実行中...
cd /d "%PROJECT_ROOT%"
start /b python "%MCP_SERVER_FILE%" > mcp_server.log 2>&1
echo MCPサーバー起動完了
timeout /t 3 /nobreak >nul
echo MCPサーバーをテスト中...
curl -X POST "http://localhost:%MCP_PORT%" -H "Content-Type: application/json" -d "{\"tool_id\": \"optimize_blend\", \"input\": {\"oils\": [{\"name\": \"Soybean\", \"cost\": 1.1, \"iodine\": 120}, {\"name\": \"Palm\", \"cost\": 0.9, \"iodine\": 80}, {\"name\": \"Rapeseed\", \"cost\": 1.0, \"iodine\": 110}], \"demand\": 1000}}"
echo.
echo テスト完了！
goto end

:end
pause