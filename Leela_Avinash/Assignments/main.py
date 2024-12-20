from speech_to_text import record_audio, transcribe_audio
from generate_response import generate_response
from text_to_speech import text_to_speech

def main():
    print("\n🛒 **Welcome to the Real-Time AI Sales Assistant!** 🛒")
    print("Say 'exit' to end the chat.\n")

    while True:
        print("You: ")
        print("Listening for your input...")
        audio_file = record_audio()
        print(audio_file)
        transcribed_text = transcribe_audio(audio_file)
        print(f"Transcribed Text: {transcribed_text}")

        if "exit" in transcribed_text.lower():
            print("Goodbye! Have a great day! 👋")
            break
        
        ai_response = generate_response(transcribed_text)
        print("\nAI Sales Assistant:", ai_response, "\n")

        text_to_speech(ai_response)

if __name__ == "__main__":
    main()
