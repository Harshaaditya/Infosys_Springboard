import sounddevice as sd
import wavio
import speech_recognition as sr

# Constants for recording
RATE = 16000
DURATION = 10  # Record for 10 seconds
FILENAME = "recorded_audio.wav"

def record_audio():
    print("Recording... Speak now!")
    audio = sd.rec(int(RATE * DURATION), samplerate=RATE, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    wavio.write(FILENAME, audio, RATE, sampwidth=2)
    print(f"Audio recorded and saved as {FILENAME}")

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            print("Transcription:", text)
            return text
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
    return None

if __name__ == "__main__":
    record_audio()
    transcribed_text = transcribe_audio(FILENAME)
