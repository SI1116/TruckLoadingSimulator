@echo off
REM ===== Streamlitアプリ起動スクリプト =====
cd /d %~dp0
echo ----------------------------------------
echo 起動中: app.py
echo ----------------------------------------

REM 仮想環境があれば有効化
if exist venv\Scripts\activate (
    call venv\Scripts\activate
)

REM Streamlitを起動
python -m streamlit run app.py

pause
