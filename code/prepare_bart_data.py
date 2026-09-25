from datasets import load_dataset
from transformers import BartTokenizer

MODEL_NAME = "facebook/bart-base"

print("Loading dataset...")

dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only actual paraphrase pairs
dataset = dataset.filter(lambda x: x["label"] == 1)

print("Loading tokenizer...")

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)


def tokenize_data(examples):
    inputs = tokenizer(
        examples["sentence1"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        text_target=examples["sentence2"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    inputs["labels"] = targets["input_ids"]

    return inputs


print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_data,
    batched=True
)

print("\nDataset ready!")
print(tokenized_dataset)

print("\nExample:")
print("Input:", dataset["train"][0]["sentence1"])
print("Target:", dataset["train"][0]["sentence2"])