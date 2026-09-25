import torch
from datasets import load_dataset
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
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
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
dataset = dataset.filter(lambda x: x["label"] == 1)

print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))
print("Test examples:", len(dataset["test"]))

# --------------------------------------------------
# 3. Load BART-base
# --------------------------------------------------

print("\nLoading BART-base...")

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)

model = BartForConditionalGeneration.from_pretrained(model_name)

# --------------------------------------------------
# 4. Tokenization
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
# 5. Data collator
# --------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# --------------------------------------------------
# 6. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./bart_full_finetuned",

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
# 7. Trainer
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
# 8. Start Full Fine-Tuning
# --------------------------------------------------

print("\n======================================")
print("STARTING FULL FINE-TUNING")
print("======================================\n")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

train_result = trainer.train()

# --------------------------------------------------
# 9. Training results
# --------------------------------------------------

print("\n======================================")
print("TRAINING COMPLETED")
print("======================================")

print("\nTraining metrics:")
print(train_result.metrics)

if torch.cuda.is_available():

    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    print(
        "\nPeak GPU memory:",
        round(peak_memory, 2),
        "GB"
    )

# --------------------------------------------------
# 10. Save final model
# --------------------------------------------------

print("\nSaving fine-tuned model...")

trainer.save_model("./bart_full_finetuned")
tokenizer.save_pretrained("./bart_full_finetuned")

print("\nModel saved to:")
print("./bart_full_finetuned")

print("\nFull fine-tuning finished successfully!")