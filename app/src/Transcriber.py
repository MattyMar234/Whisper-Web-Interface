import os
import threading
import time
from typing import Callable, List, Optional
from faster_whisper import WhisperModel
import torch
from datetime import datetime
import librosa
import whisper
from Setting import *
from dataclasses import dataclass


class Transcription:
    def __init__(self, id, display_name, language, model, created_at, folder, temperature="?"):
        self.id = id
        self.display_name = display_name
        self.language = language
        self.model = model
        self.temperature = temperature
        self.created_at = created_at
        self.folder = folder
        self.status = "completed"  # completed, error, processing
        self.file_path = self.generate_file_path()

    def __str__(self) -> str:
        return f"Transcription(id={self.id}, display_name={self.display_name}, language={self.language}, model={self.model}, created_at={self.created_at}, temperature={self.temperature}, file_path={self.file_path})"
    
    @staticmethod
    def load_transcriptions(folder:str) -> List['Transcription']:
        transcriptions = []
        
        files = os.listdir(folder)
        files.sort(key=str.lower)
        
        if files is None:
            return transcriptions
        
        for file in files:
            if file.endswith(".txt"):
                elements = [e.replace(']', '').replace('[', '') for e in file.split("]-[")]
                elements[-1] = ''.join(elements[-1].split(".txt")[0:-1])
                
                # Estrai la temperatura se presente, altrimenti usa il valore di default "?"
                temperature = "?"
                if len(elements) >= 6:
                    temperature = elements[5]
                
                if len(elements) >= 5:
                    transcription = Transcription(
                        id=elements[0],
                        display_name=elements[4],
                        language=elements[2],
                        model=elements[3],
                        created_at=elements[1],
                        folder=folder,
                        temperature=temperature
                    )
                    transcriptions.append(transcription)
        return transcriptions
    
    def generate_file_path(self) -> str:
        safe_display_name = self.display_name.replace(']', '').replace('[', '')
        return os.path.join(self.folder, f"[{self.id}]-[{self.created_at}]-[{self.language}]-[{self.model}]-[{safe_display_name}]-[{self.temperature}].txt")
    
    def get_download_name(self) -> str:
        safe_display_name = self.display_name.replace(']', '').replace('[', '')
        return f"[{safe_display_name}]-[{self.created_at}]-[{self.language}]-[{self.model}]-[t{self.temperature}].txt" 
    
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
            'temperature': self.temperature,
            'file_path': self.folder
        }

@dataclass
class QueueItem:
    id: str
    filename: str
    file_path: str
    language: str
    model_name: str
    add_info: bool = False
    vad_filter: bool = True
    beam_size: int = 5
    temperature: float = 0.0
    best_of: int = 5
    compression_ratio_threshold: float = 2.4
    no_repeat_ngram_size: int = 0
    vad_parameters: Optional[dict] = None
    patience: Optional[float] = None
    status: str = "pending"  # pending, processing, completed, error
    progress: int = 0
    created_at: Optional[str]  = None
    
    def __post_init__(self):
        if self.vad_parameters is None:
            self.vad_parameters = {"min_silence_duration_ms": 1000}
        if self.created_at is None:
            self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      
    # def __str__(self) -> str:
    #     #return f"QueueItem(id={self.id}, filename={self.filename}, language={self.language}, model={self.model_name}, status={self.status}, progress={self.progress}%), created_at={self.created_at}), filter={self.vad_filter}, beam_size={self.beam_size}), add_info={self.add_info})"
    #     return f"QueueItem(id={self.id}, filename={self.filename}, language={self.language}, model={self.model_name}, status={self.status}, progress={self.progress}%, created_at={self.created_at}, vad_filter={self.vad_filter}, beam_size={self.beam_size}, temperature={self.temperature}, best_of={self.best_of}, compression_ratio_threshold={self.compression_ratio_threshold}, no_repeat_ngram_size={self.no_repeat_ngram_size}, patience={self.patience}, add_info={self.add_info})"
        
        
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
    def __init__(self, callback: Optional[Callable] = None, workers: int = 1, cpu_threads: int = 4):
        #self.model_name = model_name
        #self.model = whisper.load_model(model_name)
        self.__current_status: str = "idle"
        self.__current_file: str = ""
        self._lock = threading.Lock()
        self._callback: Optional[Callable] = callback
        self._stop_flag: bool = False
        self._current_device: Optional[str] = None
        self.__workers: int = workers
        self.__cpu_threads: int = cpu_threads
        
        torch.set_float32_matmul_precision("high")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
    def getCurrentFile(self) -> str:
        return self.__current_file
    
    def getCurrentStatus(self) -> str:
        return self.__current_status
    
    def stop_transcription(self):
        """Imposta il flag per fermare l'esecuzione della trascrizione corrente."""
        #with self._lock:
        self._stop_flag = True
    
    def get_current_device(self) -> Optional[str]:
        """Restituisce il device su cui sta venendo eseguito il modello o None se non è in esecuzione."""
        
        if self.__current_status == "idle":
            return None 
        return self._current_device
       
    def __format_time(self, seconds) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"    
    
    def transcribe(self, queueLock, item: QueueItem, updateFunc: Callable) -> Transcription:
        
        # Resetta il flag di stop all'inizio della trascrizione
        with self._lock:
            self._stop_flag = False
            self._current_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.__current_file = item.filename
        
            transcription = Transcription(
                id=item.id,
                display_name=item.filename,
                language=item.language,
                model=item.model_name,
                created_at=item.created_at,
                folder=TRANSCRIPTIONS_DIR,
                temperature=str(item.temperature)
            ) 
            
        transcription.status = "processing"
        total_duration = librosa.get_duration(path=item.file_path)
        logger.info(f"Audio duration: {self.__format_time(total_duration)}") 
        logger.info(f"Current transcription: {transcription}")
        output_path = transcription.file_path
        
        try: 
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")
                
            # with queueLock:
            #     if item.status == "removed":
            #         logger.info(f"File removed.")
            #         # pulizia stato locale
            #         with self._lock:
            #             self.__current_file = ""
            #             self.__current_status = "idle"
                        
            #         if updateFunc:
            #             try:
            #                 updateFunc()   # chiamata fuori dal lock
            #             except Exception:
            #                 logger.exception("updateFunc raised an exception")

                    
            #         return transcription
                
                item.status = "processing"
                self.__current_status = "processing"
                
            
            #https://developer.nvidia.com/rdp/cudnn-archive
            model = WhisperModel(
                model_size_or_path=item.model_name,
                device=self._current_device,
                device_index=0,
                #compute_type="float16" if torch.cuda.is_available() else "default",
                cpu_threads=self.__cpu_threads,
                num_workers=self.__workers
            )
            
            segments, info = model.transcribe(
                item.file_path,
                language=item.language if item.language and item.language != "auto" else None,
                task="transcribe",
                beam_size=item.beam_size,
                vad_filter=item.vad_filter,
                vad_parameters=item.vad_parameters,
                temperature=[item.temperature],
                # best_of=item.best_of,
                compression_ratio_threshold=item.compression_ratio_threshold,
                no_repeat_ngram_size=item.no_repeat_ngram_size,
                # patience=item.patience if item.patience is not None else 1,
            )
            #print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

            last_int_progress_percent = -1
            last_update_time = time.time()
            dt = 0.5  # intervallo minimo tra gli aggiornamenti in secondi
            
            with open(output_path, "a", encoding="utf-8") as f:
                for segment in segments:
                    
                    # check stop
                    if self._stop_flag:
                        with self._lock:
                            logger.info("Transcriber stopped!")
                            self.__current_status = "stopped"
                            transcription.status = "stopped"
                            break
                   
                        
                    
                    #Calcola la percentuale di completamento in base alla durata totale
                    progress_percent = (segment.end / total_duration) * 100 if total_duration > 0 else 0
                    int_progress_percent = min(100, int(progress_percent))
                    
                    logger.info(f"[{item.filename}] Segment {segment.start:.2f}s to {segment.end:.2f}s: {segment.text} (Progress: {progress_percent:.3f}%)")
                    
                    # if int_progress_percent > last_int_progress_percent:
                    #     if time.time() - last_update_time >= dt: 
                    #         last_update_time = time.time()
                            
                    #         with self._lock:
                    #             last_int_progress_percent = int_progress_percent
                    #             item.progress = int_progress_percent
                    #             if updateFunc:
                    #                 updateFunc()
                    
                    # if item.add_info:
                    #     # Formatta l'output con timestamp in formato HH:MM:SS
                    #     segmentrange = f"[{self.__format_time(segment.start)} -> {self.__format_time(segment.end)}]"
                    #     progress_info = f"[Progress: {progress_percent:.3f}%]"
                    #     data = f"{segmentrange} {progress_info} "
                    #     fixed_data = f"{data:<45}"
                    #     text = f"{fixed_data}: {segment.text}"
                    #     f.write(text + "\n")
                    # else:
                    #     f.write(segment.text + "\n")

                    # aggiorna progress brevemente sotto lock, ma CALLBACK fuori dal lock
                    call_update = False
                    with self._lock:
                        if int_progress_percent > last_int_progress_percent and (time.time() - last_update_time >= dt):
                            last_int_progress_percent = int_progress_percent
                            item.progress = int_progress_percent
                            last_update_time = time.time()
                            call_update = True

                    if call_update and updateFunc:
                        try:
                            updateFunc()   # chiamata fuori dal lock
                        except Exception:
                            logger.exception("updateFunc raised an exception")

                    # scrivi testo (IO) — non serve lock
                    if item.add_info:
                        segmentrange = f"[{self.__format_time(segment.start)} -> {self.__format_time(segment.end)}]"
                        progress_info = f"[Progress: {progress_percent:.3f}%]"
                        data = f"{segmentrange} {progress_info} "
                        fixed_data = f"{data:<45}"
                        text = f"{fixed_data}: {segment.text}"
                        f.write(text + "\n")
                    else:
                        f.write(segment.text + "\n")
                    
                self.__current_status = "completed"
            
            with self._lock:
                self.__current_status = "completed"
                transcription.status = "completed"      
            
            if self._callback is not None:
                try:
                    self._callback()
                except Exception:
                    logger.exception("callback raised an exception")
                
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            with self._lock:
                self.__current_status = "error"
                transcription.status = "error"
            if updateFunc:
                try:
                    updateFunc()
                except Exception:
                    logger.exception("updateFunc raised unhandled exception in error path")
        finally:
            # pulizia stato in ogni caso
            with self._lock:
                self.__current_file = ""
                if self.__current_status == "processing":
                    # se non è stato settato a stopped/completed/error, impostalo idle
                    self.__current_status = "idle"
            if updateFunc:
                try:
                    updateFunc()
                except Exception:
                    logger.exception("updateFunc raised an exception")

            return transcription