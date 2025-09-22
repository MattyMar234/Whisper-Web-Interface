import os
import threading
import time
from typing import Callable, List
from faster_whisper import WhisperModel
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datetime import datetime
import librosa
import whisper
from Setting import *


class Transcription:
    def __init__(self, id, display_name, language, model, created_at, folder):
        self.id = id
        self.display_name = display_name
        self.language = language
        self.model = model
        self.created_at = created_at
        self.folder = folder
        self.status = "completed"  # completed, error, processing
        self.file_path = self.generate_file_path()

    def __str__(self) -> str:
        return f"Transcription(id={self.id}, display_name={self.display_name}, language={self.language}, model={self.model}, created_at={self.created_at}, file_path={self.file_path})"

    @staticmethod
    def load_transcriptions(folder:str) -> list:
        transcriptions = []
        
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                elements = [e.replace(']', '').replace('[', '') for e in file.split("]-[")]
                elements[-1] = ''.join(elements[-1].split(".txt")[0:-1])
                
                if len(elements) >= 5:
                    transcription = Transcription(
                        id= elements[0],
                        display_name=elements[4],
                        language=elements[2],
                        model=elements[3],
                        created_at=elements[1],
                        folder=folder
                    )
                    transcriptions.append(transcription)
                
    
        return transcriptions
    
    def generate_file_path(self) -> str:
        safe_display_name = self.display_name.replace(']', '').replace('[', '')
        return os.path.join(self.folder, f"[{self.id}]-[{self.created_at}]-[{self.language}]-[{self.model}]-[{safe_display_name}].txt")
        
    
    def rename(self, new_display_name):
        self.display_name = new_display_name
        new_path = self.generate_file_path()
        os.rename(self.file_path, new_path)
        self.file_path = new_path
        
    def to_dict(self):
        return {
            'id': self.id,
            'display_name': self.display_name,
            'language': self.language,
            'model': self.model,
            'created_at': self.created_at,
            'file_path': self.folder
        }

class QueueItem:
    def __init__(self, id, filename, file_path, language, model_name, add_info=False, 
                 vad_filter=True, beam_size=5, temperature=0.0, best_of=5, 
                 compression_ratio_threshold=2.4, no_repeat_ngram_size=0, 
                 vad_parameters=None, patience=None):
        self.id = id
        self.filename = filename
        self.file_path = file_path
        self.language = language
        self.model_name = model_name
        self.add_info = add_info
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        self.temperature = temperature
        self.best_of = best_of
        self.compression_ratio_threshold = compression_ratio_threshold
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.vad_parameters = vad_parameters or {"min_silence_duration_ms": 1000}
        self.patience = patience
        self.status = "pending"  # pending, processing, completed, error
        self.progress = 0
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      
    def __str__(self) -> str:
        #return f"QueueItem(id={self.id}, filename={self.filename}, language={self.language}, model={self.model_name}, status={self.status}, progress={self.progress}%), created_at={self.created_at}), filter={self.vad_filter}, beam_size={self.beam_size}), add_info={self.add_info})"
        return f"QueueItem(id={self.id}, filename={self.filename}, language={self.language}, model={self.model_name}, status={self.status}, progress={self.progress}%, created_at={self.created_at}, vad_filter={self.vad_filter}, beam_size={self.beam_size}, temperature={self.temperature}, best_of={self.best_of}, compression_ratio_threshold={self.compression_ratio_threshold}, no_repeat_ngram_size={self.no_repeat_ngram_size}, patience={self.patience}, add_info={self.add_info})"
        
        
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
    
    def transcribe(self, item: QueueItem, updateFunc: Callable) -> Transcription:
        

    
        total_duration = librosa.get_duration(path=item.file_path)
        #print(f"Audio duration: {format_time(total_duration)}") 
        

        
        transcription = Transcription(
            id=item.id,
            display_name=item.filename,
            language=item.language,
            model=item.model_name,
            created_at=item.created_at,
            folder=TRANSCRIPTIONS_DIR,
        ) 
        
        output_path = transcription.file_path
        
        print(transcription) 
        
        try: 
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")
        
           
            self.current_status = "processing"
            
            #https://developer.nvidia.com/rdp/cudnn-archive
            model = WhisperModel(
                model_size_or_path=item.model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                device_index=0,
                #compute_type="float16" if torch.cuda.is_available() else "default",
                cpu_threads=4,
                num_workers=1
            )
            
            
                
                
              
    
            
            with self.lock:
                segments, info = model.transcribe(
                    item.file_path,
                    language=item.language if item.language and item.language != "auto" else None,
                    task="transcribe",
                    beam_size=item.beam_size,
                    vad_filter=item.vad_filter,
                    vad_parameters=item.vad_parameters,
                    temperature=[item.temperature],
                    best_of=item.best_of,
                    compression_ratio_threshold=item.compression_ratio_threshold,
                    no_repeat_ngram_size=item.no_repeat_ngram_size,
                    patience=item.patience if item.patience is not None else 1,
                )
            #print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            # Passa il parametro della lingua solo se è specificato e non è "none"

            
            last_int_progress_percent = -1
            last_update_time = time.time()
            dt = 0.5  # intervallo minimo tra gli aggiornamenti in secondi
            
            with open(output_path, "a", encoding="utf-8") as f:

                for segment in segments:
                    # Calcola la percentuale di completamento in base alla durata totale
                    progress_percent = (segment.end / total_duration) * 100 if total_duration > 0 else 0
                    int_progress_percent = min(100, int(progress_percent))
                    
                    logger.info(f"[{item.filename}] Segment {segment.start:.2f}s to {segment.end:.2f}s: {segment.text} (Progress: {progress_percent:.3f}%)")
                    
                    if int_progress_percent > last_int_progress_percent:
                        if time.time() - last_update_time >= dt: 
                            last_update_time = time.time()
                            
                            with self.lock:
                                last_int_progress_percent = int_progress_percent
                                item.progress = int_progress_percent
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
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            self.current_status = "error"
            transcription.status = "error"
            if updateFunc:
                updateFunc()
       
        return transcription