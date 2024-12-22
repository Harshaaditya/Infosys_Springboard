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

OUTPUT_FILE = "output_audio.wav"

def get_text_to_speech_client():
    return build("texttospeech", "v1", developerKey=api_key)

def synthesize_speech(text, client, output_file=OUTPUT_FILE, language_code="en-US", ssml_gender="FEMALE"):
    print("Synthesizing speech...")

    request_payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": language_code,
            "ssmlGender": ssml_gender,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "speakingRate": 1.0,  
            "pitch": 3.0  
        },
    }

    try:
        response = client.text().synthesize(body=request_payload).execute()

        if "audioContent" in response:
            audio_content = base64.b64decode(response["audioContent"])

            with wave.open(output_file, "wb") as wf:
                wf.setnchannels(1) 
                wf.setsampwidth(2) 
                wf.setframerate(16000)
                wf.writeframes(audio_content)
            print(f"Audio saved to: {output_file}")
        else:
            print("No audio content received.")

    except googleapiclient.errors.HttpError as err:
        print(f"Error during synthesis: {err}")

def play_audio(file_path):
    print("Playing audio...")
    try:
        wf = wave.open(file_path, "rb")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return
    except wave.Error as e:
        print(f"Error reading audio file: {e}")
        return

    chunk = 1024
    p = pyaudio.PyAudio()

    try:
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(chunk)
        while data:
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        print("Audio playback finished.")
    except Exception as e:
        print(f"Error during audio playback: {e}")
    finally:
        wf.close()
        p.terminate()

def text_to_speech(text):
    tts_client = get_text_to_speech_client()
    synthesize_speech(text, tts_client)
    play_audio(OUTPUT_FILE)



def audio_record():
    speech_client = get_speech_client()
    audio_file = record_audio_with_silence()
    transcribed_text = transcribe_audio_google(audio_file, speech_client)
    return transcribed_text
