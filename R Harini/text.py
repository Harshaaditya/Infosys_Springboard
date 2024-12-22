import google.generativeai as Gen_Ai

api_key = "AIzaSyAeooMuSJF8RuZJDawtRGYYp-DzdTJxaZ0"
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
def chatbot_response(inp):
    chat = model.start_chat()
    response = chat.send_message(inp)
    return response.text
while True:
    inp = input("You: ")
    if inp.lower() in ['exit', 'quit']:
        print("Exiting the chatbot.")
        break
    response = chatbot_response(inp)
    print("Chatbot:", response)
