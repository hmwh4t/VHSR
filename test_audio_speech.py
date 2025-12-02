# Load model directly
from faster_whisper import WhisperModel
import pyaudio
import wave
import threading
import numpy as np
import torch

# Initialize Faster Whisper Model
# Use "medium" or "large-v2" depending on your VRAM. "medium" is a good balance.
# compute_type="float16" requires GPU. Use "int8" or "float32" for CPU.
MODEL_PATH = "PhoWhisper-medium-ct2"
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel(MODEL_PATH, device=device, compute_type=compute_type)

def transcribe_audio(filename):
    # Transcribe using Faster Whisper
    # beam_size=5 is standard.
    segments, info = model.transcribe(filename, beam_size=5, language="vi", task="transcribe")
    
    # Collect text from segments
    transcription = "".join([segment.text for segment in segments])
    return transcription

if __name__ == "__main__":
    audio_file = "recorded_audio.wav"
    print("Transcribing...")
    try:
        text = transcribe_audio(audio_file)
        print("-" * 20)
        print("Transcript:")
        print(text)
        print("-" * 20)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
