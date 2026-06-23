@echo off
echo ========================================================
echo Arret des serveurs precedents (ports 8000 et 8501)...
echo ========================================================
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8501 "') do taskkill /F /PID %%a 2>nul
timeout /t 2 /nobreak > NUL

echo ========================================================
echo  Bike Sharing Demand - Lancement en local
echo ========================================================

echo 1. Lancement de l'API FastAPI sur le port 8000...
start cmd /k "py -m uvicorn app:app --port 8000"

timeout /t 3 /nobreak > NUL

echo 2. Lancement du Dashboard Streamlit...
start cmd /k "py -m streamlit run streamlit_app.py"

echo C'est fait ! Vous pouvez fermer cette fenetre.
exit