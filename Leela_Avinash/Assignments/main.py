from MileStone_1.speech_to_text import record_audio, transcribe_audio
from MileStone_1.generate_response import generate_response
from MileStone_1.text_to_speech import text_to_speech
# from MileStone_2.Analyze_user_statement import Analyze_text
from MileStone_2.Analyze_user_audio import analyze_audio
import time

def main():
    print("\n🛒 **Welcome to the Real-Time AI Sales Assistant!** 🛒")
    print("Say 'exit' to end the chat.\n")

    while True:
        print("You: ")
        print("Listening for your input...")
        audio_file = record_audio()
        start = time.time()
        print(audio_file)
        transcription_start = time.time()
        transcribed_text = transcribe_audio(audio_file)
        print(f"Transcribed Text: {transcribed_text}")
        print("Time taken for transcription:", time.time() - transcription_start)

        if "exit" in transcribed_text.lower():
            transcribed_text = "Goodbye! Have a great day!"
            print(transcribed_text)
            text_to_speech(transcribed_text)
            break
        summary_start = time.time()
        # summary = Analyze_text(transcribed_text)
        summary = analyze_audio(audio_file)
        print(summary)
        print(f"Time taken for summary: {time.time() - summary_start:.2f} seconds")
        response_start = time.time()
        ai_response = generate_response(transcribed_text, summary)
        print("\nAI Sales Assistant:", ai_response, "\n")
        print(f"Time taken for response generation: {time.time() - response_start:.2f} seconds")
        print(f"Total Time taken: {time.time() - start:.2f} seconds")

        text_to_speech(ai_response)

if __name__ == "__main__":
    main()
