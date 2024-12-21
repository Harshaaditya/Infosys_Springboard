import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
checkpoint_dir = "fine_tuned_gpt2"
model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_dir)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Input prompt
prompt = input("Enter the prompt: ")

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate text with adjusted parameters
output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=500,  # Allow for more text to be generated
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
)

# Decode output and truncate at the first full stop
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
if '.' in decoded_output:
    truncated_output = decoded_output[:decoded_output.index('.') + 1]
else:
    truncated_output = decoded_output  # Fallback if no full stop is found

# Ensure only English characters are kept
english_output = ''.join(char for char in truncated_output if char.isascii())
print(english_output)
