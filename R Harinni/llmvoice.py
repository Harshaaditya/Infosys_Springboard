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

# Text-to-Speech Function
def text_to_speech(text, output_audio_file="output.mp3"):
    tts = gTTS(text, lang="en")
    tts.save(output_audio_file)
    print(f"Audio saved as {output_audio_file}")
    os.system(f"start {output_audio_file}")

# Speech-to-Text Function (Extended Listening with Silence Detection)
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Speak now. Wait 5 seconds after speaking to stop.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            # Listen with a timeout and phrase limit for silence detection
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Could not request results from Google Speech Recognition service.")
            return None

# Text Generation Function (Using Fine-Tuned GPT-2)
def generate_text(prompt, model, tokenizer, max_length=150):
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        do_sample=True,
    )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    # Clean the output to include only English characters and valid punctuation
    cleaned_output = ''.join(c for c in decoded_output if c.isalnum() or c in " .,!?")
    return cleaned_output

# Main function that integrates everything
def voice_based_llm():
    while True:
        print("Please speak to provide input for generating text... (say 'finish' to end)")
        prompt = speech_to_text()
        
        if not prompt:
            print("No input detected, please try again.")
            continue
        
        if "finish" in prompt.lower():
            print("Ending the program.")
            text_to_speech("The program is now ending. Goodbye!")
            break
        
        print(f"Generated prompt: {prompt}")
        
        # Generate text from the model
        generated_text = generate_text(prompt, model, tokenizer, max_length=150)
        print(f"Generated Text: {generated_text}")
        
        # Convert generated text to speech and play audio
        text_to_speech(generated_text)

if __name__ == "__main__":
    voice_based_llm()
