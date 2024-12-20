import pyaudio
import wave
import numpy as np
import time
from googleapiclient.discovery import build
import googleapiclient.errors
import base64
import json
from dotenv import load_dotenv
import os
from gtts import gTTS
from playsound import playsound

load_dotenv()
api_key = os.getenv("SPEECH_TO_TEXT_API")
print(api_key)

RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
SILENCE_THRESHOLD = 500  
SILENCE_DURATION = 4  
FILENAME = "temp_recording.wav"
TTS_OUTPUT_FILE = "output_audio.mp3"  

def get_speech_client():
    return build("speech", "v1", developerKey=api_key)

def is_silent(audio_chunk):
    return np.max(np.abs(audio_chunk)) < SILENCE_THRESHOLD

def record_audio_with_silence():
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
        print("\nRecording stopped manually.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    print("Audio recording saved as:", FILENAME)
    return FILENAME

def transcribe_audio_google(file_path, speech_client):
    print("Transcribing...")

    start_time = time.time()

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
        encoded_audio = base64.b64encode(content).decode("utf-8")

    request_payload = {
        "config": {
            "encoding": "LINEAR16",
            "languageCode": "en-US",
            "sampleRateHertz": RATE,
        },
        "audio": {
            "content": encoded_audio
        },
    }

    try:
        response = speech_client.speech().recognize(body=request_payload).execute()

        end_time = time.time()
        transcription_time = end_time - start_time
        print(f"Time taken to transcribe: {transcription_time:.2f} seconds")

        if "results" in response:
            transcript = " ".join(
                result["alternatives"][0]["transcript"] for result in response["results"]
            )
            print("\nTranscription:", transcript)
            return transcript
        else:
            print("No transcription results.")
            return ""

    except googleapiclient.errors.HttpError as err:
        print(f"Error during transcription: {err}")
        return ""

def text_to_speech(text, output_file):
    if not text.strip():
        print("No text to convert to speech.")
        return

    print("Converting text to speech...")
    tts = gTTS(text=text, lang="en")
    tts.save(output_file)
    print(f"Speech saved to {output_file}")
    play_audio(output_file)

def play_audio(file_path):
    print("Playing audio...")
    playsound(file_path)

if __name__ == "__main__":
    speech_client = get_speech_client()
    audio_file = record_audio_with_silence()
    transcribed_text = transcribe_audio_google(audio_file, speech_client)

    if transcribed_text:
        text_to_speech(transcribed_text, TTS_OUTPUT_FILE)
