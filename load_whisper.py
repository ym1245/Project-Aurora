from faster_whisper import WhisperModel
from io import BytesIO
import os

class STTModel:
    def __init__(self,model_path="https://huggingface.co/ghost613/faster-whisper-large-v3-turbo-korean",device="auto",compute_type="default"):
        self.model=WhisperModel(model_path,device=device,compute_type=compute_type)

    def transcribe(self,audio_path,language="ko",**kwargs):
        if isinstance(audio_path,BytesIO):
            segments,info=self.model.transcribe(audio_path,language=language,**kwargs)
        else:
            if not os.path.exists(audio_path):
                raise FileNotFoundError
            segments,info=self.model.transcribe(audio_path,language=language,**kwargs)
        result={
            "text":" ".join(segment.text for segment in segments),
            "language":info.language,
            "language_probaility":info.language_probability,
            "duration":info.duration,
        }
        return result

def load_model(model_path):
    return STTModel(model_path)
