# Usa Python slim image
FROM python:3.12-slim

# Imposta variabili d'ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV WHISPER_MODEL=medium

# Installa dipendenze di sistema
RUN apt-get update 
RUN apt-get install -y openssh-server
RUN apt-get install -y ffmpeg
RUN apt-get install -y wget
RUN apt-get install -y curl
#RUN apt-get install -y git
RUN rm -rf /var/lib/apt/lists/*

# Configura SSH
RUN mkdir -p /var/run/sshd
RUN echo 'root:1234' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# RUN wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb
# RUN dpkg -i cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb
# RUN cp /var/cudnn-local-repo-ubuntu2204-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/
# RUN apt-get update -y
# RUN apt-get -y install libcudnn9-cuda-12 libcudnn9-dev-cuda-12

RUN wget https://developer.download.nvidia.com/compute/cudnn/9.1.0/local_installers/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
RUN dpkg -i cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb
RUN cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update -y
RUN apt-get install -y libcudnn9-cuda-12=9.1.0.* libcudnn9-dev-cuda-12=9.1.0.*
RUN apt-get install -y libcudnn9-samples=9.1.0.*

# Installa Python packages
RUN pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
 

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
RUN echo '#!/bin/bash\n\
# Avvia SSH in background\n\
/usr/sbin/sshd -D -e &\n\
# Avvia applicazione\n\
python3 /app/src/main.py' > /start.sh && chmod +x /start.sh

# Imposto lo script come entrypoint
ENTRYPOINT ["/start.sh"]

# Comando di avvio
#CMD ["/start.sh"]