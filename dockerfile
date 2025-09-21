# Usa Python slim image
FROM python:3.12-slim

# Imposta variabili d'ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV WHISPER_MODEL=medium

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    openssh-server 
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configura SSH
RUN mkdir -p /var/run/sshd
RUN echo 'root:1234' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Installa Python packages
# RUN pip install --no-cache-dir \
#     flask \
#     openai-whisper \
#     torch \
#     torchaudio \
#     --index-url https://download.pytorch.org/whl/cpu

# Crea directory app
WORKDIR /app

# Copia codice applicazione
COPY ./app/ .


# Installa dipendenze Python aggiuntive
RUN pip install -r requirements.txt

# Espone le porte
EXPOSE 12345
EXPOSE 12346

# Script di avvio
# RUN echo '#!/bin/bash\n\
# # Avvia SSH in background\n\
# /usr/sbin/sshd -D -e &\n\
# # Avvia applicazione\n\
# python app.py' > /start.sh && chmod +x /start.sh

RUN echo '#!/bin/bash\n\
# Avvia SSH in background\n\
/usr/sbin/sshd -D -e' > /start.sh && chmod +x /start.sh

# Comando di avvio
CMD ["/start.sh"]