import speech_recognition as sr
from gtts import gTTS
import os
import playsound

def recognize_speech():
    """
    Recognizes speech from the microphone and converts it to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None

def text_to_speech(text):
    """
    Converts text to speech and plays the audio.
    """
    tts = gTTS(text=text, lang='en')
    audio_file = "response.mp3"
    tts.save(audio_file)
    playsound.playsound(audio_file)
    os.remove(audio_file)