import torch
from transformers import BartTokenizer, BartForConditionalGeneration

MODEL_PATH = "./bart_full_finetuned_improved"

print("Loading tokenizer...")
tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")
model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("\nModel loaded successfully!")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\n--- BART Text Paraphrasing ---")

while True:
    sentence = input("\nEnter a sentence (or type 'exit' to quit): ")

    if sentence.lower() == "exit":
        break

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    paraphrase = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    print("\nInput:     ", sentence)
    print("Paraphrase:", paraphrase)