from assignment_01 import record_audio, transcribe_audio
from assignment_02 import text_to_speech
from assignment_03 import chat_session  # Import the chat session from assignment_03

def main():
    print("\n*Welcome to the Real-Time AI Sales Assistant!*\n")
    print("Say 'exit' to end the chat.\n")
    
    while True:
        print("You: ")
        print("Listening for your input...")
        
        # Record and transcribe user input
        record_audio()  # Record the audio
        transcribed_text = transcribe_audio("recorded_audio.wav")  # Transcribe the audio
        
        if transcribed_text:
            print(f"You said: {transcribed_text}")
            
            # Check if user wants to exit
            if "exit" in transcribed_text.lower():
                goodbye_message = "Goodbye! Have a great day!"
                print(f"AI Sales Assistant: {goodbye_message}")
                text_to_speech(goodbye_message)
                break
            
            # Generate AI response using the Generative AI model
            try:
                ai_response = chat_session.send_message(transcribed_text).text
                print(f"\nAI Sales Assistant: {ai_response}\n")
                
                # Convert AI response to speech
                text_to_speech(ai_response)
            except Exception as e:
                print(f"Error generating AI response: {e}")
                text_to_speech("Sorry, I couldn't process your request. Please try again.")
        else:
            print("Sorry, I didn't catch that. Please try again.")

if __name__ == "__main__":
    main()
