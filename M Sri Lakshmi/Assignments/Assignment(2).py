import google.generativeai as genai
import whisper
import pyttsx3
import pyaudio
import wave
import numpy as np
import time
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Retrieve the Gemini API key from the environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Whisper model
model = whisper.load_model("base")

# Initialize TTS (Pyttsx3)
engine = pyttsx3.init()

# Set up Gemini API
genai.configure(api_key=gemini_api_key)  # Use the API key from the .env file
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# Audio recording settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500  # Amplitude threshold for silence
SILENCE_DURATION = 2  # Duration of silence in seconds to stop recording
audio = pyaudio.PyAudio()

# Function to detect silence
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
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()

    # Save as WAV
    temp_file = "temp_audio.wav"
    with wave.open(temp_file, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    # Transcribe using Whisper
    try:
        print("Transcribing...")
        result = model.transcribe(temp_file)
        return result['text']
    except Exception as e:
        print(f"Error during transcription: {e}")
        return "Error: Could not transcribe the audio."

# Query Gemini API
def query_gemini(prompt):
    print("Sending query to Gemini LLM...")
    messages = [{"parts": [prompt], "role": "user"}]
    try:
        response = gemini_model.generate_content(contents=messages)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return "Error: No valid response from Gemini."
    except Exception as e:
        print(f"Error generating content from Gemini: {e}")
        return "Error: Could not generate a response from Gemini."

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
            print("You asked:", query)

            print("Generating response...")
            answer = query_gemini(query)
            print("AI Response:", answer)

            print("Speaking response...")
            speak_text(answer)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print("Error:", e)
    finally:
        audio.terminate()
