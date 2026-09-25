import torch
from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate

MODEL_NAME = "facebook/bart-base"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()

# Load PAWS-X
dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only actual paraphrase pairs
test_data = dataset["test"].filter(lambda x: x["label"] == 1)

print("Test examples:", len(test_data))

predictions = []
references = []

# Generate predictions
for i in range(0, len(test_data), 4):

    batch = test_data[i:i + 4]

    inputs = tokenizer(
        batch["sentence1"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4
        )

    generated = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    predictions.extend(generated)
    references.extend(batch["sentence2"])

    if (i + 4) % 100 == 0:
        print(f"Processed {min(i + 4, len(test_data))}/{len(test_data)}")

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# BLEU
bleu_result = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

# ROUGE
rouge_result = rouge.compute(
    predictions=predictions,
    references=references
)

print("\n===== BASELINE RESULTS =====")

print("BLEU:", round(bleu_result["bleu"], 4))
print("ROUGE-1:", round(rouge_result["rouge1"], 4))
print("ROUGE-2:", round(rouge_result["rouge2"], 4))
print("ROUGE-L:", round(rouge_result["rougeL"], 4))

print("\n===== EXAMPLES =====")

for i in range(5):
    print("\nOriginal:")
    print(test_data[i]["sentence1"])

    print("Reference:")
    print(test_data[i]["sentence2"])

    print("Generated:")
    print(predictions[i])