import google.generativeai as Gen_Ai
import speech_recognition as sr
import pyttsx3

api_key = "Google-Api"

Gen_Ai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.7, 
    "top_p": 0.95,
    "top_k": 40, 
    "max_output_tokens": 8000,
    "response_mime_type": "text/plain",
}

model = Gen_Ai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def chatbot_response(inp):
    chat = model.start_chat()
    short_answer_prompt = f"Please give a brief answer: {inp}"
    response = chat.send_message(short_answer_prompt)
    
    cleaned_response = response.text.replace("*", "").strip()
    
    return cleaned_response

def listen_to_audio():
    with sr.Microphone() as source:
        print("Listening for your message...")
        audio = recognizer.listen(source)
        try:
            message = recognizer.recognize_google(audio)
            print(f"You said: {message}")
            return message
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return ""

def speak_response(response):
    engine.say(response)
    engine.runAndWait()

while True:
    print("Say 'exit' or 'quit' to end.")
    
    user_input = listen_to_audio()
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chatbot.")
        speak_response("Goodbye!")
        break
    
    chatbot_output = chatbot_response(user_input)
    print("Chatbot:", chatbot_output)
    speak_response(chatbot_output)
