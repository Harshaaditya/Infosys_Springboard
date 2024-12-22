import google.generativeai as Gen_Ai
import speech_recognition as sr
import pyttsx3

api_key = "AIzaSyAeooMuSJF8RuZJDawtRGYYp-DzdTJxaZ0"

Gen_Ai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.7, 
    "top_p": 0.95,
    "top_k": 40, 
    "max_output_tokens": 250,  
    "response_mime_type": "text/plain",
}

model = Gen_Ai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

recognizer = sr.Recognizer()
engine = pyttsx3.init()

conversation_history = []

def chatbot_response(user_input):
    conversation_history.append(f"User: {user_input}")
    prompt = (
        "You are a Real-Time AI Sales Assistant, designed to help customers with their inquiries and sales decisions.\n"
        "Respond in a helpful, clear, and persuasive manner, focusing on customer needs, product benefits, and guiding them towards a purchase decision.\n"
        "Avoid lengthy explanations and be short and clear."
        "- Answer in short and make sure to be specific to the product enquiry"
        "- It can only answer questions related to any product which is in the market"
        "- Acknowledge the user's concern."
        "- Focus on key product benefits and features."
        "- Reassure with proof, examples, or guarantees."
        "- Provide an actionable suggestion, like scheduling a demo or offering a discount."
        f"Conversation History:{''.join([f'{entry}' for entry in conversation_history])}"
        f"User: {user_input}AI Sales Assistant:"
    )
    
    chat = model.start_chat()
    response = chat.send_message(prompt)
    answer = response.text.strip().replace('*', '')  
    conversation_history.append(f"AI Sales Assistant: {answer}")
    
    return answer

def listen_to_audio():
    with sr.Microphone() as source:
        print("Listening for customer message...")
        audio = recognizer.listen(source)
        try:
            message = recognizer.recognize_google(audio)
            print(f"Customer: {message}")
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

def generate_deal_suggestion(user_input):
    if "price" in user_input.lower():
        return "You might want to offer a special discount or flexible payment plan."
    elif "features" in user_input.lower():
        return "Emphasize the exclusive features of the product that are perfect for their needs."
    elif "contract" in user_input.lower():
        return "Suggest signing an agreement with a limited-time offer to encourage a faster decision."
    else:
        return "Offer to answer any questions or clarify details about the product."

while True:
    print("Say 'exit' or 'quit' to end.")
    
    user_input = listen_to_audio()
    if user_input.lower() in ['exit', 'quit']:
        print("Ending conversation.")
        speak_response("Goodbye!")
        break
    
    chatbot_output = chatbot_response(user_input)
    print("AI Sales Assistant:", chatbot_output)
    speak_response(chatbot_output)
    
    negotiation_suggestion = generate_deal_suggestion(user_input)
    print("Negotiation Tip:", negotiation_suggestion)
    speak_response(negotiation_suggestion)
