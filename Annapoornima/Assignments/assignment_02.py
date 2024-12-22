import pyttsx3

def text_to_speech(text):
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Set properties for the voice (optional)
    engine.setProperty('rate', 150)  # Speed of speech (words per minute)
    engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
    
    # List available voices (optional)
    voices = engine.getProperty('voices')
    for index, voice in enumerate(voices):
        print(f"Voice {index}: {voice.name}")
    
    # Choose a specific voice (optional, replace '0' with another index if needed)
    engine.setProperty('voice', voices[0].id)
    
    # Speak the text
    print("Speaking...")
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    # Example text input
    text = input("Enter the text to convert to speech: ")
    if text.strip():
        text_to_speech(text)
    else:
        print("No text provided!")
