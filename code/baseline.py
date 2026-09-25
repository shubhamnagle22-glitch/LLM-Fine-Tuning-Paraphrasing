import torch
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-base"

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Input sentence
text = "The student completed the assignment successfully."

# Tokenize
inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=128,
    truncation=True
).to(device)

# Generate paraphrase
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=128,
        num_beams=5,
        early_stopping=True
    )

# Decode output
paraphrase = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("Original:")
print(text)

print("\nGenerated:")
print(paraphrase)