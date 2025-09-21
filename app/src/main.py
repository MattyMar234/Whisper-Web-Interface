import os
import tempfile
import threading
import time
from typing import Callable, List
import uuid
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import logging

from app.src.Transcriber import Transcription
from app.src.Transcriber import QueueItem, Transcriber
from app.src.Setting import *


# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebServer:
    def __init__(self, host='0.0.0.0', port=12345):
        
        
        self._modelName = 'small'
        
        self._Transcriber = Transcriber(model_name=self._modelName)
        
        #queue per l'elaborazione in background
        self._queueLock = threading.Lock()
        self._queue: List[QueueItem] = []
        self._maxQueue = 5
        
        # Avvia il thread di elaborazione
        self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()
        
        
        # Memoria delle trascrizioni
        self._transcriptions = {"test": Transcription("ad", "da", "fe", "fef", "ad", "", "")}
        
        
        self._app: Flask = Flask(__name__)
        self._app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
        
        self._socketio = SocketIO(self._app, cors_allowed_origins="*")
        
        self._app.route('/', methods=['GET'])(self.index)
        self._app.route('/transcribe', methods=['POST'])(self.transcribe)
        self._app.route('/transcription', methods=['GET'])(self.get_transcriptions)
        self._app.route('/transcription/<trans_id>', methods=['GET'])(self.get_transcription)
        self._app.route('/transcription/<trans_id>', methods=['PUT'])(self.rename_transcription)
        self._app.route('/transcription/<trans_id>', methods=['DELETE'])(self.delete_transcription)
        self._app.route('/transcription/<trans_id>/download', methods=['GET'])(self.download_transcription)
        self._app.route('/health', methods=['GET'])(self.health_check)
        
        # Eventi SocketIO
        self._socketio.on('connect')(self._handle_connect)
        self._socketio.on('disconnect')(self._handle_disconnect)
        self._socketio.on('get_queue_status')(self._send_queue_status)
        self._socketio.on('get_transcriptions')(self._send_transcriptions)
        
        self._socketio.run(self._app, host=host, port=port, debug=True)
    
    def _handle_connect(self):
        logger.info("Client connesso")
        self._send_queue_status()
        self._send_transcriptions()
        
    def _handle_disconnect(self):
        logger.info("Client disconnesso")
        
    def _send_queue_status(self):
        with self._queueLock:
            queue_status = [item.to_dict() for item in self._queue]
        
        self._socketio.emit('queue_status', {
            'queue': queue_status,
            'transcriber_status': self._Transcriber.current_status,
            'current_file': self._Transcriber.current_file
        })
        
    def _send_transcriptions(self):
        transcriptions = [t.to_dict() for t in self._transcriptions.values()]
        self._socketio.emit('transcriptions_update', {'transcriptions': transcriptions})
            
        
    def load_available_transcriptions(self):
        for filename in os.listdir(TRANSCRIPTIONS_DIR):
            if filename.endswith(".txt"):
                data = filename
    
    def allowed_file(self, filename) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS   


    def on_transcription_complete(self, **data):
        pass
    
    
    def _process_queue(self):
        while True:
            time.sleep(2)
            
            with self._queueLock:
                if self._queue:
                    item = self._queue[0]
                    item.status = "processing"
                    self._Transcriber.current_file = item.filename
                else:
                    continue
            
            print("set status processing")
            self._send_queue_status()
            
            if item is not None and isinstance(item, QueueItem):
                try:
                    # Processa il file
                    output_file = self._Transcriber.transcribe(
                        item, updateFunc=lambda: self._send_queue_status()
                    )
                    
                    if True:
                        # Salva la trascrizione
                
                        #file_path = os.path.join(TRANSCRIPTIONS_DIR, f"{item.id}.txt")
                        
                        # with open(file_path, 'w', encoding='utf-8') as f:
                        #     f.write(text)
                        
                        # Aggiungi alla memoria
                        transcription = Transcription(
                            id=item.id,
                            filename=item.filename,
                            display_name=item.filename,
                            language=item.language,
                            model=item.model_name,
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            file_path=output_file,
                            #status="completed"
                        )
                        
                        print(item.id, output_file)
                        
                        self._transcriptions[item.id] = transcription
                        self._send_transcriptions()
                        
                        # Aggiorna lo stato della coda
                        item.status = "completed"
                        item.progress = 100
                        self._send_queue_status()
                
                except Exception as e:
                    logger.error(f"Errore nell'elaborazione del file {item.filename}: {str(e)}")
                    item.status = "error"
                    self._send_queue_status()
                
                # Rimuovi il file temporaneo
                try:
                    os.remove(item.file_path)
                except:
                    pass
                
                self._Transcriber.current_file = None
                self._Transcriber.current_status = "idle"
                self._send_queue_status()
            
            else:
                item.status = "error"
                self._send_queue_status()
            
            with self._queueLock:
                if self._queue:
                    self._queue.remove(item)
                    
            self._send_queue_status()
            time.sleep(1)  # Attendi prima di controllare di nuovo la coda

    

    def transcribe(self):
        # Verifica presenza file
        if 'files' not in request.files:
            return jsonify({"error": "Nessun file fornito"}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({"error": "Nessun file selezionato"}), 400



        # Parametri opzionali
        language = request.form.get('language', None)
        model_name = request.form.get('model', None)
        add_info = request.form.get('add_info', 'false').lower() == 'true'
        vad_filter = request.form.get('vad_filter', 'true').lower() == 'true'
        beam_size = int(request.form.get('beam_size', 5))
        
        results = []
      

        # Processa i file in parallelo
        with self._queueLock:
            
            if len(self._queue) + len(files) > self._maxQueue:
                logger.error(f"Coda piena.")
                return jsonify({
                    "success": False,
                    "error": f"Coda piena. Massimo {self._maxQueue} file contemporaneamente."
                }), 429
                
            
            for file in files:
                if file and self.allowed_file(file.filename) and file.filename is not None:
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(self._app.config['UPLOAD_FOLDER'], filename)

                    try:
                        file.save(temp_path)
                        logger.info(f"File salvato temporaneamente in {temp_path}")
                        #self._queue.append(temp_path)
                        
                        # Aggiungi alla coda
                        item_id = str(uuid.uuid4())
                        item = QueueItem(
                            id=item_id,
                            filename=filename,
                            file_path=temp_path,
                            language=language,
                            model_name=model_name,
                            add_info=add_info,
                            vad_filter=vad_filter,
                            beam_size=beam_size
                        )
                        
                        self._queue.append(item)
                        results.append({
                            "id": item_id,
                            "filename": filename,
                            "success": True
                        })
                        
                    except Exception as e:
                        logger.error(f"Errore salvataggio file {filename}: {str(e)}")
                        results.append({
                            "filename": filename,
                            "success": False,
                            "error": f"Errore salvataggio: {str(e)}"
                        })
                
        # Notifica i client
        self._send_queue_status()  
                                   
        return jsonify({
            "success": True,
            "results": results
        })
        

    def get_transcriptions(self):
        return jsonify([t.to_dict() for t in self._transcriptions.values()])

    def get_transcription(self, trans_id):
        if trans_id in self._transcriptions:
            trans = self._transcriptions[trans_id]
            try:
                with open(trans.file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return jsonify({
                    'id': trans_id,
                    'filename': trans.filename,
                    'display_name': trans.display_name,
                    'text': text,
                    'language': trans.language,
                    'model': trans.model,
                    'created_at': trans.created_at,
                    'status': trans.status
                })
            except Exception as e:
                return jsonify({"error": f"Errore lettura file: {str(e)}"}), 500
        return jsonify({"error": "Trascrizione non trovata"}), 404

    def rename_transcription(self, trans_id):
        data = request.get_json()
        if not data or 'display_name' not in data:
            return jsonify({"error": "Nome non specificato"}), 400
        
        if trans_id in self._transcriptions:
            self._transcriptions[trans_id].display_name = data['display_name']
            self._send_transcriptions()
            return jsonify({"success": True, "display_name": data['display_name']})
        
        return jsonify({"error": "Trascrizione non trovata"}), 404
    
    def index(self):
        return render_template(
            'index.html', 
            languages=SUPPORTED_LANGUAGES,
            models=SUPPORTED_MODELS,
            transcriptions= [t.to_dict() for t in self._transcriptions.values()] 
        )

    def delete_transcription(self, trans_id):
        if trans_id in self._transcriptions:
            trans = self._transcriptions[trans_id]
            try:
                os.remove(trans.file_path)
            except:
                pass
            del self._transcriptions[trans_id]
            self._send_transcriptions()
            return jsonify({"success": True})
        
        return jsonify({"error": "Trascrizione non trovata"}), 404

    def download_transcription(self, trans_id):
        
        print(self._transcriptions.keys())
        print(trans_id)
        
        if trans_id in self._transcriptions:
            trans = self._transcriptions[trans_id]
            try:
                return send_file(
                    trans.file_path,
                    as_attachment=True,
                    download_name=f"{trans.file_path.split("/")[-1]}",#f"{trans.display_name}.txt",
                    mimetype='text/plain'
                )
            except Exception as e:
                return jsonify({"error": f"Errore download: {str(e)}"}), 500
        
        return jsonify({"error": "Trascrizione non trovata"}), 404
        
    def health_check(self):
        return jsonify({"status": "healthy", "model": self._modelName})
 

def main():
    wb = WebServer()



if __name__ == "__main__":
    main()