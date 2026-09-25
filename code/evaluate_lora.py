import torch
import numpy as np

from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from peft import PeftModel

import evaluate


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Load PAWS-X test dataset
# --------------------------------------------------

print("\nLoading PAWS-X test dataset...")

dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only actual paraphrase pairs
test_dataset = dataset["test"].filter(
    lambda x: x["label"] == 1
)

print("Test paraphrase examples:", len(test_dataset))


# --------------------------------------------------
# 3. Load BART-base + LoRA adapter
# --------------------------------------------------

print("\nLoading BART-base...")

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)

base_model = BartForConditionalGeneration.from_pretrained(
    model_name
)

print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    "./bart_lora"
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# 4. Load evaluation metrics
# --------------------------------------------------

print("\nLoading evaluation metrics...")

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

batch_size = 4

all_predictions = []
all_references = []

print("\nGenerating LoRA predictions...\n")


for start in range(0, len(test_dataset), batch_size):

    batch = test_dataset[
        start:min(start + batch_size, len(test_dataset))
    ]

    inputs = tokenizer(
        batch["sentence1"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
        )

    predictions = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    all_predictions.extend(predictions)

    all_references.extend(
        batch["sentence2"]
    )

    if (start + batch_size) % 100 == 0:
        print(
            f"Processed {min(start + batch_size, len(test_dataset))}"
            f"/{len(test_dataset)}"
        )


# --------------------------------------------------
# 6. Calculate BLEU
# --------------------------------------------------

print("\nCalculating BLEU...")

bleu_result = bleu_metric.compute(
    predictions=all_predictions,
    references=[
        [reference]
        for reference in all_references
    ]
)


# --------------------------------------------------
# 7. Calculate ROUGE
# --------------------------------------------------

print("Calculating ROUGE...")

rouge_result = rouge_metric.compute(
    predictions=all_predictions,
    references=all_references
)


# --------------------------------------------------
# Calculate BERTScore
# --------------------------------------------------

print("Calculating BERTScore...")

bertscore_metric = evaluate.load("bertscore")

bertscore_result = bertscore_metric.compute(
    predictions=all_predictions,
    references=all_references,
    lang="en",
    device=device
)

bert_precision = np.mean(bertscore_result["precision"])
bert_recall = np.mean(bertscore_result["recall"])
bert_f1 = np.mean(bertscore_result["f1"])


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n======================================")
print("LoRA EVALUATION RESULTS")
print("======================================")

print(
    f"BLEU:      {bleu_result['bleu']:.4f}"
)

print(
    f"ROUGE-1:   {rouge_result['rouge1']:.4f}"
)

print(
    f"ROUGE-2:   {rouge_result['rouge2']:.4f}"
)

print(
    f"ROUGE-L:   {rouge_result['rougeL']:.4f}"
)

print(
    f"BERTScore Precision: {bert_precision:.4f}"
)

print(
    f"BERTScore Recall:    {bert_recall:.4f}"
)

print(
    f"BERTScore F1:        {bert_f1:.4f}"
)

# --------------------------------------------------
# 9. Show qualitative examples
# --------------------------------------------------

print("\n======================================")
print("QUALITATIVE EXAMPLES")
print("======================================")

for i in range(min(10, len(all_predictions))):

    print(f"\nExample {i + 1}")

    print("Original :", test_dataset[i]["sentence1"])

    print("Reference:", all_references[i])

    print("Generated :", all_predictions[i])


print("\nLoRA evaluation completed successfully!")