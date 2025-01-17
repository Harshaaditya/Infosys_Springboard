import os
import time
import wave
import numpy as np
import pyaudio
import whisper
import pyttsx3
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configure Whisper and Pyttsx3
model = whisper.load_model("base")
engine = pyttsx3.init()

# Configure Gemini
genai.configure(api_key=gemini_api_key)
gemini_model = genai.GenerativeModel("gemini-2.0-flash-exp")

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2
audio = pyaudio.PyAudio()

# Detect silence in audio
def is_silent(data_chunk):
    return np.max(np.abs(data_chunk)) < SILENCE_THRESHOLD

# Record and transcribe audio
def record_and_transcribe():
    print("Listening... Speak now.")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
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
    finally:
        stream.stop_stream()
        stream.close()

    temp_file = "temp_audio.wav"
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    try:
        print("Transcribing...")
        result = model.transcribe(temp_file)
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error: Could not transcribe the audio."

# Analyze text using Gemini
def analyze_text(text):
    analysis_prompt = f"""
    Analyze the following text and provide only the tone, sentiment, and intent as the result.
    Text: "{text}"
    Provide the output in this structured format:
    - Sentiment: [Positive/Negative/Neutral]
    - Tone: [Detected tones]
    - Intent: [User's intent]
    """
    try:
        response = gemini_model.generate_content(analysis_prompt)
        return response.text
    except Exception as e:
        print(f"Error during text analysis with Gemini: {e}")
        return "Error: Could not analyze the text using Gemini."

# Convert text to speech
def speak_text(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

# Main process
if __name__ == "__main__":
    try:
        while True:
            print("Start speaking or press Ctrl+C to exit.")
            query = record_and_transcribe()
            print("You said:", query)

            print("Analyzing text...")
            analysis_report = analyze_text(query)
            print("\n### Analysis Result ###")
            print(analysis_report)

            print("Speaking response...")
            speak_text(analysis_report)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        audio.terminate()