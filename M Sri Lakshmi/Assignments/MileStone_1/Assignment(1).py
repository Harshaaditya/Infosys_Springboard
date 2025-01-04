import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is available
if not gemini_api_key:
    raise ValueError("API key not found. Please ensure GEMINI_API_KEY is set in the .env file.")

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)

# Create a model instance
model = genai.GenerativeModel("gemini-1.5-flash")

# Send a prompt to the model
prompt = "What is AI?"
response = model.generate_content(prompt)

# Print the response
print(response.text)
