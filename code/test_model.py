import torch
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-base"

print("Loading tokenizer...")
tokenizer = BartTokenizer.from_pretrained(model_name)

print("Loading model...")
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("\nModel loaded successfully!")
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("GPU memory allocated:",
      round(torch.cuda.memory_allocated(0) / 1024**2, 2),
      "MB" if torch.cuda.is_available() else "")