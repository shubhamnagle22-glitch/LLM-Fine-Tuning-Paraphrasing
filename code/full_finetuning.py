import torch
import re

from datasets import load_dataset, concatenate_datasets

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU Memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )


# --------------------------------------------------
# 2. Load PAWS-X dataset
# --------------------------------------------------

print("\nLoading PAWS-X dataset...")

dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only paraphrase pairs
dataset = dataset.filter(
    lambda x: x["label"] == 1
)

print("\nOriginal dataset sizes:")
print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print("Test examples:", len(dataset["test"]))


# --------------------------------------------------
# 3. Filter highly similar paraphrase pairs
# --------------------------------------------------

def word_overlap(sentence1, sentence2):

    words1 = set(
        re.findall(r"\b\w+\b", sentence1.lower())
    )

    words2 = set(
        re.findall(r"\b\w+\b", sentence2.lower())
    )

    if not words1 or not words2:
        return 1.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


original_train_size = len(dataset["train"])

dataset["train"] = dataset["train"].filter(
    lambda x: word_overlap(
        x["sentence1"],
        x["sentence2"]
    ) < 0.90
)

print(
    "\nTraining examples after overlap filtering:",
    len(dataset["train"])
)

print(
    "Training examples removed:",
    original_train_size - len(dataset["train"])
)


# --------------------------------------------------
# 4. Create bidirectional paraphrase pairs
# --------------------------------------------------

def create_reverse_pair(example):

    return {
        "sentence1": example["sentence2"],
        "sentence2": example["sentence1"],
        "label": example["label"],
    }


print("\nCreating reverse paraphrase pairs...")

reverse_train = dataset["train"].map(
    create_reverse_pair
)

dataset["train"] = concatenate_datasets(
    [
        dataset["train"],
        reverse_train
    ]
)

print(
    "Training examples after bidirectional augmentation:",
    len(dataset["train"])
)

print(
    "Validation examples:",
    len(dataset["validation"])
)

print(
    "Test examples:",
    len(dataset["test"])
)


# --------------------------------------------------
# 5. Load BART-base
# --------------------------------------------------

print("\nLoading BART-base...")

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(
    model_name
)

model = BartForConditionalGeneration.from_pretrained(
    model_name
)


# --------------------------------------------------
# 6. Tokenization
# --------------------------------------------------

def tokenize_function(examples):

    inputs = tokenizer(
        examples["sentence1"],
        max_length=128,
        truncation=True,
    )

    targets = tokenizer(
        text_target=examples["sentence2"],
        max_length=128,
        truncation=True,
    )

    inputs["labels"] = targets["input_ids"]

    return inputs


print("\nTokenizing dataset...")

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# --------------------------------------------------
# 7. Data collator
# --------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# --------------------------------------------------
# 8. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./bart_full_finetuned_improved",

    # Training
    num_train_epochs=1,
    learning_rate=5e-5,

    # RTX 3050 6GB - conservative configuration
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,

    # Memory optimisation
    fp16=True,

    # Evaluation
    eval_strategy="epoch",

    # Save model
    save_strategy="epoch",
    save_total_limit=1,

    # Logging
    logging_steps=100,

    # Reproducibility
    seed=42,

    # Disable external reporting
    report_to="none",
)


# --------------------------------------------------
# 9. Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],

    processing_class=tokenizer,
    data_collator=data_collator,
)


# --------------------------------------------------
# 10. Start Full Fine-Tuning
# --------------------------------------------------

print("\n======================================")
print("STARTING IMPROVED FULL FINE-TUNING")
print("======================================\n")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


train_result = trainer.train()


# --------------------------------------------------
# 11. Training results
# --------------------------------------------------

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print("\nTraining metrics:")
print(train_result.metrics)


if torch.cuda.is_available():

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        "\nPeak GPU memory:",
        round(peak_memory, 2),
        "GB"
    )


# --------------------------------------------------
# 12. Save final model
# --------------------------------------------------

print("\nSaving improved fine-tuned model...")

trainer.save_model(
    "./bart_full_finetuned_improved"
)

tokenizer.save_pretrained(
    "./bart_full_finetuned_improved"
)

print("\nModel saved to:")
print("./bart_full_finetuned_improved")

print("\nImproved full fine-tuning finished successfully!")