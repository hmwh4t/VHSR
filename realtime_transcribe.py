"""
Real-time Vietnamese Speech Recognition using Chunkformer-CTC-Large-Vie
with dynamic noise calibration and VAD-based transcription.
"""

import time
import threading
import numpy as np
import pyaudio
import wave
import tempfile
import os
from chunkformer import ChunkFormerModel
import torch

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = "khanhld/chunkformer-ctc-large-vie"
LANGUAGE = "vi"
RATE = 16000
CHANNELS = 1
CHUNK = 1024
INPUT_DEVICE_INDEX = None  # None uses default. Set to integer index (e.g., 0, 1, 2) if default text is silent.
MAX_RECORDING_DURATION = 30  # Maximum seconds before forced transcription
SILENCE_THRESHOLD = 0.01     # Base amplitude threshold (dynamically adjusted)
SILENCE_DURATION = 2.0       # Seconds of silence to trigger transcription
NOISE_CALIBRATION_DURATION = 2.0  # Seconds to record for noise calibration

# =============================================================================
# Initialize Model
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading ChunkFormer model '{MODEL_PATH}' on {device}...")
model = ChunkFormerModel.from_pretrained(MODEL_PATH)
if device == "cuda":
    model = model.cuda()
print("Model loaded.")

# =============================================================================
# Global State
# =============================================================================
audio_buffer = np.array([], dtype=np.float32)
recording = True
speech_active = False
lock = threading.Lock()
last_speech_time = time.time()


def list_audio_devices():
    """List available audio input devices to help select the correct index."""
    p = pyaudio.PyAudio()
    try:
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        
        print("\nAvailable Audio Input Devices:")
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                name = p.get_device_info_by_host_api_device_index(0, i).get('name')
                print(f"Index {i}: {name}")
        print("------------------------------------------------\n")
    except Exception as e:
        print(f"Error listing audio devices: {e}")
    finally:
        p.terminate()


def calibrate_noise():
    """
    Record background noise for calibration and dynamically adjust
    the silence threshold to be above the noise floor.
    """
    global SILENCE_THRESHOLD
    
    print(f"Calibrating background noise... Please remain silent for {NOISE_CALIBRATION_DURATION} seconds.")
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    num_chunks = int(RATE * NOISE_CALIBRATION_DURATION / CHUNK)
    
    for _ in range(num_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        frames.append(chunk)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    noise_audio = np.concatenate(frames)
    noise_rms = np.sqrt(np.mean(noise_audio ** 2))
    
    if noise_rms == 0.0:
        print("WARNING: Background noise RMS is 0.0. The microphone might be muted or the wrong device index is selected.")

    # Set threshold to 2x the noise floor (minimum of base threshold)
    new_threshold = max(SILENCE_THRESHOLD, noise_rms * 2.0)
    SILENCE_THRESHOLD = new_threshold
    
    print(f"Calibration complete. Noise RMS: {noise_rms:.5f}, Silence Threshold: {SILENCE_THRESHOLD:.5f}")


def record_audio():
    """
    Continuously record audio and detect speech/silence transitions.
    Only buffers audio when speech is detected.
    """
    global audio_buffer, last_speech_time, speech_active
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK
    )
    
    print("Recording started... Press Ctrl+C to stop.")
    
    while recording:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate RMS for silence detection
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            
            if rms > SILENCE_THRESHOLD:
                last_speech_time = time.time()
                if not speech_active:
                    speech_active = True
                    print("\n[Speech detected] Recording...")
            
            if speech_active:
                with lock:
                    audio_buffer = np.concatenate((audio_buffer, audio_chunk))
                
                # Check if silence threshold exceeded
                if time.time() - last_speech_time > SILENCE_DURATION:
                    speech_active = False
                    print("[Silence detected] Processing...")
        
        except Exception as e:
            print(f"Recording error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Recording stopped.")


def transcribe_loop():
    """
    Monitor the audio buffer and transcribe when:
    1. Speech ends (silence detected after speech)
    2. Max recording duration is reached
    """
    global audio_buffer
    
    start_time = time.time()
    
    while recording:
        current_time = time.time()
        
        with lock:
            buffer_duration = len(audio_buffer) / RATE
        
        should_transcribe = False
        
        # Transcribe if max duration reached
        if buffer_duration >= MAX_RECORDING_DURATION:
            should_transcribe = True
            print("\n[Max duration reached] Processing...")
        # Transcribe if speech ended and we have audio
        elif not speech_active and buffer_duration > 0:
            should_transcribe = True
        
        if should_transcribe:
            with lock:
                audio_to_process = audio_buffer.copy()
                audio_buffer = np.array([], dtype=np.float32)
            
            if len(audio_to_process) > 0:
                temp_filename = None
                try:
                    # Save buffer to temp file as ChunkFormer expects a file path
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_filename = f.name
                        with wave.open(temp_filename, 'wb') as wf:
                            wf.setnchannels(CHANNELS)
                            wf.setsampwidth(2) # 16-bit
                            wf.setframerate(RATE)
                            # Convert float32 [-1, 1] to int16
                            audio_int16 = (audio_to_process * 32767).astype(np.int16)
                            wf.writeframes(audio_int16.tobytes())
                    
                    # Call ChunkFormer
                    full_text = model.endless_decode(
                        audio_path=temp_filename,
                        chunk_size=64,
                        left_context_size=128,
                        right_context_size=128,
                        total_batch_duration=14400,
                        return_timestamps=False
                    )
                    
                    if str(full_text).strip():
                        elapsed = int(current_time - start_time)
                        print(f"[{elapsed}s]: {full_text}")

                except Exception as e:
                    print(f"Transcription error: {e}")
                finally:
                    # Clean up temporary file
                    if temp_filename and os.path.exists(temp_filename):
                        os.remove(temp_filename)
        
        time.sleep(0.1)


def main():
    global recording
    
    # List devices so user knows what indices are available
    list_audio_devices()
    
    # Calibrate noise threshold
    calibrate_noise()
    
    # Start recording in background thread
    record_thread = threading.Thread(target=record_audio, daemon=True)
    record_thread.start()
    
    # Run transcription loop in main thread
    try:
        transcribe_loop()
    except KeyboardInterrupt:
        print("\nStopping...")
        recording = False
        record_thread.join(timeout=2.0)
        print("Exited.")


if __name__ == "__main__":
    main()
