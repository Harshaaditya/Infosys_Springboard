import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from gtts import gTTS
import speech_recognition as sr

# Load the fine-tuned model and tokenizer
checkpoint_dir = "fine_tuned_gpt2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)

# Speech-to-Text Function (Continuous Listening)
def speech_to_text_continuous():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Ask your sales question, say 'finish' to stop.")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        while True:
            try:
                audio = recognizer.listen(source)
                print("Recognizing...")
                text = recognizer.recognize_google(audio)
                print(f"Detected: {text}")

                # Check if the user says "finish"
                if "finish" in text.lower():
                    print("Stopping...")
                    return "finish"
                else:
                    # Return detected text to pass to the model for generation
                    return text
            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service.")
                break

# Text Generation Function (Using Fine-Tuned GPT-2 for Sales Q&A)
def generate_sales_response(question, model, tokenizer, max_length=150):
    model.eval()

    # Explicitly ask for a sales-related answer
    prompt = f"Provide a clear and concise sales answer to the following question:\n\nQuestion: {question}\nAnswer:"

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.9,  # Reduce randomness for more focused answers
        temperature=0.6,  # Lower temperature to reduce randomness
        do_sample=True,
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the answer part by trimming the question portion
    start_index = decoded_output.lower().find("answer:") + len("answer:")
    answer = decoded_output[start_index:].strip()

    # Clean the output to include only English characters and valid punctuation
    cleaned_output = ''.join(c for c in answer if c.isalnum() or c in " .,!?")
    return cleaned_output

# Text-to-Speech Function
def text_to_speech(text, output_audio_file="output.mp3"):
    tts = gTTS(text, lang="en")
    tts.save(output_audio_file)
    print(f"Audio saved as {output_audio_file}")
    os.system(f"start {output_audio_file}")  # Automatically play on Windows

# Main function that integrates everything
def voice_based_sales_interaction():
    while True:
        print("Please ask your sales question...")
        question = speech_to_text_continuous()
        
        if question == "finish":
            print("Ending the program.")
            break
        
        print(f"Question received: {question}")
        
        # Generate sales response from the model
        generated_answer = generate_sales_response(question, model, tokenizer, max_length=150)
        print(f"Generated Sales Answer: {generated_answer}")
        
        # Convert generated sales answer to speech
        text_to_speech(generated_answer)

if __name__ == "__main__":
    voice_based_sales_interaction()
