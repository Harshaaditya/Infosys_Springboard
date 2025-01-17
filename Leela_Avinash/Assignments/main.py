import csv
from MileStone_1.speech_to_text import record_audio, transcribe_audio
from MileStone_1.generate_response import generate_response
from MileStone_1.text_to_speech import text_to_speech
from MileStone_2.Analyze_user_audio import analyze_audio
from MileStone_2.Analyze_user_statement import Analyze_text
from MileStone_3.Reccomendations import recommend
from MileStone_3.PostCallAnalysis import generate_summary
import time

def read_csv_content(csv_file):
    """Reads the content of the CSV file and returns a consolidated text."""
    conversation = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) == 2: 
                conversation.append(f"User: {row[0]} AI: {row[1]}")
    return " ".join(conversation)

def main():
    print("\n🛒 **Welcome to the Real-Time AI Sales Assistant!** 🛒")
    print("Say 'exit' to end the chat.\n")

    csv_file = "conversation_log.csv"

    # Clear the file content and write header
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writerow(["User Input", "AI Response"])  # Write the header row

    # Append new conversation
    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_ALL)

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
                ai_response = "Goodbye! Have a great day!"
                print(ai_response)
                text_to_speech(ai_response)
                writer.writerow([transcribed_text, ai_response])
                full_conversation = read_csv_content(csv_file)
                analysis = Analyze_text(full_conversation)
                post_call_analysis = generate_summary(full_conversation, analysis)
                print("Post Call Analysis")
                print(post_call_analysis)
                break

            summary_start = time.time()
            summary = analyze_audio(audio_file)
            print(summary)
            print(f"Time taken for summary: {time.time() - summary_start:.2f} seconds")
            
            recommended_terms = recommend(1, transcribed_text, summary)
            response_start = time.time()
            ai_response = generate_response(transcribed_text, summary, recommended_terms)
            print("\nAI Sales Assistant:", ai_response, "\n")
            print(f"Time taken for response generation: {time.time() - response_start:.2f} seconds")
            print(f"Total Time taken: {time.time() - start:.2f} seconds")
            
            text_to_speech(ai_response)

            writer.writerow([transcribed_text, ai_response])

if __name__ == "__main__":
    main()
 