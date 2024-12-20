import pyaudio
import wave
import numpy as np
import whisper
import time

RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD = 500 
SILENCE_DURATION = 2 

def is_silent(data_chunk):
    """Check if the chunk of audio is silent based on a threshold."""
    return np.max(np.abs(data_chunk)) < SILENCE_THRESHOLD

def record_audio():
    """Record audio from the microphone until the user stops talking."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening... Speak now.")
    frames = []
    silence_start_time = None

    try:
        while True:
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)
            frames.append(data)

            if is_silent(audio_chunk):
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_DURATION:
                    print("Silence detected. Stopping recording.")
                    break
            else:
                silence_start_time = None

    except KeyboardInterrupt:
        print("\nRecording stopped by user.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    filename = "temp_recording.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    return filename


def transcribe_audio(file_path):
    """Transcribe the recorded audio using the Whisper model."""
    model = whisper.load_model("small")
    print("Transcribing...")
    result = model.transcribe(file_path)
    return result["text"]