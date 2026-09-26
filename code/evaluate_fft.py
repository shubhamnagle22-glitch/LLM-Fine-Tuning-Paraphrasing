import torch
from datasets import load_dataset
from transformers import BartForConditionalGeneration, BartTokenizer
import evaluate

# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 2. Load improved fine-tuned model
# --------------------------------------------------

model_path = "./bart_full_finetuned_improved"

print("\nLoading improved fine-tuned BART model...")

tokenizer = BartTokenizer.from_pretrained(model_path)

model = BartForConditionalGeneration.from_pretrained(
    model_path
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# 3. Load test dataset
# --------------------------------------------------

print("\nLoading PAWS-X test dataset...")

dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only paraphrase pairs
test_data = dataset["test"].filter(
    lambda x: x["label"] == 1
)

print("Test examples:", len(test_data))


# --------------------------------------------------
# 4. Generate paraphrases
# --------------------------------------------------

predictions = []
references = []

batch_size = 4

print("\nGenerating paraphrases...")

for i in range(0, len(test_data), batch_size):

    batch = test_data[i:i + batch_size]

    inputs = tokenizer(
        batch["sentence1"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
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
            early_stopping=True
        )

    generated = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    predictions.extend(generated)
    references.extend(batch["sentence2"])

    if (i + batch_size) % 100 < batch_size:
        print(
            f"Processed {min(i + batch_size, len(test_data))}"
            f"/{len(test_data)}"
        )


# --------------------------------------------------
# 5. Calculate BLEU
# --------------------------------------------------

print("\nCalculating metrics...")

bleu = evaluate.load("bleu")

bleu_result = bleu.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)


# --------------------------------------------------
# 6. Calculate ROUGE
# --------------------------------------------------

rouge = evaluate.load("rouge")

rouge_result = rouge.compute(
    predictions=predictions,
    references=references
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n======================================")
print("IMPROVED FULL FINE-TUNING RESULTS")
print("======================================")

print(
    "BLEU:",
    round(bleu_result["bleu"], 4)
)

print(
    "ROUGE-1:",
    round(rouge_result["rouge1"], 4)
)

print(
    "ROUGE-2:",
    round(rouge_result["rouge2"], 4)
)

print(
    "ROUGE-L:",
    round(rouge_result["rougeL"], 4)
)


# --------------------------------------------------
# 8. Show examples
# --------------------------------------------------

print("\n======================================")
print("SAMPLE PREDICTIONS")
print("======================================")

for i in range(10):

    print("\nExample", i + 1)

    print(
        "Original :",
        test_data[i]["sentence1"]
    )

    print(
        "Reference:",
        references[i]
    )

    print(
        "Generated:",
        predictions[i]
    )