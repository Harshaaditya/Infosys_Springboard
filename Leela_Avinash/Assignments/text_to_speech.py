import base64
import wave
from googleapiclient.discovery import build
import googleapiclient.errors
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_CLOUD_TTS_API")

if not api_key:
    raise ValueError("API_KEY is missing. Please set it in the .env file.")

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
            "audioEncoding": "MP3",
            "speakingRate": 0.9,  
            "pitch": 2.0      
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

def text_to_speech(text):
    tts_client = get_text_to_speech_client()
    synthesize_speech(text, tts_client)