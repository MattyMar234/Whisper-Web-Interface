@echo off
setlocal

REM Nome dell'immagine
set IMAGE_NAME=pytorch-whisper-webserver
set CONTAINER_NAME=Pytorch-Whisper-WebServer
set DOCKERFILE=dockerfile

REM Imposta variabile DISPLAY per Windows con VcXsrv attivo
set DISPLAY=host.docker.internal:0.0



REM === CONTROLLO ESISTENZA DOCKERFILE ===
if not exist %DOCKERFILE% (
    echo [ERRORE] Nessun file Dockerfile trovato nella directory corrente.
    pause
    exit /b 1
)

REM === CONTROLLA SE L'IMMAGINE ESISTE ===
docker image inspect %IMAGE_NAME% >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Immagine non trovata. Avvio build...
    docker build --no-cache -t %IMAGE_NAME% -f %DOCKERFILE% .
) else (
    echo [INFO] Immagine trovata: %IMAGE_NAME%
)

REM === RIMUOVI IL CONTAINER ESISTENTE (SE PRESENTE) === ??
docker rm -f %CONTAINER_NAME% >nul 2>&1



REM Avvia il container con GPU, volume, supporto GUI e porta SSH
docker run -it ^
  --gpus all ^
  --name %CONTAINER_NAME% ^
  --shm-size=1g ^
  -e DISPLAY=%DISPLAY% ^
  -e XDG_RUNTIME_DIR=/tmp/runtime ^
  -e SDL_AUDIODRIVER=dummy ^
  -v /tmp/.X11-unix:/tmp/.X11-unix ^
  -v "%cd%":/app ^
  -p 12346:22 ^
  -p 12345:12345 ^
  -w /app ^
  %IMAGE_NAME%

REM docker exec %CONTAINER_NAME% bash

endlocal
