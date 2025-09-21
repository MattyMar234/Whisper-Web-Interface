import os
import threading
import time
from typing import Callable
from faster_whisper import WhisperModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datetime import datetime
import librosa
import whisper

from app.src.Setting import TRANSCRIPTIONS_DIR

class Transcription:
    def __init__(self, id, filename, display_name, language, model, created_at, file_path):
        self.id = id
        self.filename = filename
        self.display_name = display_name
        self.language = language
        self.model = model
        self.created_at = created_at
        self.file_path = file_path
        self.status = "completed"  # completed, error, processing
       
        
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'display_name': self.display_name,
            'language': self.language,
            'model': self.model,
            'created_at': self.created_at,
            'file_path': self.file_path
        }

class QueueItem:
    def __init__(self, id, filename, file_path, language, model_name, add_info=False, vad_filter=True, beam_size=5):
        self.id = id
        self.filename = filename
        self.file_path = file_path
        self.language = language
        self.model_name = model_name
        self.add_info = add_info
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.status = "pending"  # pending, processing, completed, error
        self.progress = 0
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'language': self.language,
            'model': self.model_name,
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at
        }


class Transcriber:
    def __init__(self, model_name='small', callback: Callable | None = None):
        self.model_name = model_name
        self.model = whisper.load_model(model_name)
        self.current_status = "idle"
        self.current_file = None
        self.lock = threading.Lock()
        self._callback = callback
        
        torch.set_float32_matmul_precision("high")
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    def format_time(self, seconds) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"    
    
    def transcribe(self, item: QueueItem, updateFunc: Callable) -> str:
        

        options = {
            "fp16": torch.cuda.is_available(),  # Disabilita fp16 per compatibilità CPU
            "language": item.language if item.language and item.language != "auto" else "none",
            "task": "transcribe",
            "beam_size": item.beam_size,
            "vad_filter": item.vad_filter,
            "temperature" : 0.0,
            "best_of": item.beam_size
        }
        
        
        total_duration = librosa.get_duration(path=item.file_path)
        #print(f"Audio duration: {format_time(total_duration)}") 
        
            
        
        with self.lock:
            self.current_status = "processing"
            
            #https://developer.nvidia.com/rdp/cudnn-archive
            model = WhisperModel(
                model_size_or_path=item.model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                device_index=0,
                compute_type="float16" if torch.cuda.is_available() else "default",
                cpu_threads=4,
                num_workers=1
            )
            
            segments, info = model.transcribe(item.file_path, **options)
            #print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            # Passa il parametro della lingua solo se è specificato e non è "none"

            output_path = os.path.join(TRANSCRIPTIONS_DIR, \
            f"[{item.id}]-[{item.created_at}]-[{info.language}]-[{item.model_name}]-[{item.filename.replace(']','').replace('[','')}].txt")
        

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")
            
            with open(output_path, "a", encoding="utf-8") as f:

                for segment in segments:
                    # Calcola la percentuale di completamento in base alla durata totale
                    progress_percent = (segment.end / total_duration) * 100 if total_duration > 0 else 0
                    item.progress = int(progress_percent)
                    
                    if updateFunc:
                        updateFunc()
                    
                    if item.add_info:
                        # Formatta l'output con timestamp in formato HH:MM:SS
                        segmentrange = f"[{self.format_time(segment.start)} -> {self.format_time(segment.end)}]"
                        progress_info = f"[Progress: {progress_percent:.3f}%]"
                        data = f"{segmentrange} {progress_info} "
                        fixed_data = f"{data:<45}"
                        text = f"{fixed_data}: {segment.text}"
                        f.write(text + "\n")
                    else:
                        f.write(segment.text + "\n")
    
                self.current_status = "completed"
                
        if self._callback is not None:
            self._callback()
        
        return output_path