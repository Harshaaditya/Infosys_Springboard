import os
import google.generativeai as Gen_Ai
Gen_Ai.configure(api_key=os.environ["Gemini_API"])
generation_config = {
    "temperature": 0.9, 
    "top_p": 0.95,
    "top_k": 40, 
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = Gen_Ai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)
chat = model.start_chat()

response = chat.send_message("what is python?")
print(response.text)