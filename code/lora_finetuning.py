import torch
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model


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

# Keep only actual paraphrase pairs
dataset = dataset.filter(
    lambda x: x["label"] == 1
)

print(
    "Training examples:",
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
# 3. Load BART-base
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
# 4. Configure LoRA
# --------------------------------------------------

print("\nConfiguring LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    bias="none",

    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(
    model,
    lora_config
)

print("\nLoRA trainable parameters:")

model.print_trainable_parameters()


# --------------------------------------------------
# 5. Tokenization
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
# 6. Data collator
# --------------------------------------------------

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)


# --------------------------------------------------
# 7. Training configuration
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./bart_lora",

    num_train_epochs=1,

    learning_rate=5e-5,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    fp16=True,

    eval_strategy="epoch",

    save_strategy="epoch",
    save_total_limit=1,

    logging_steps=100,

    seed=42,

    report_to="none",
)


# --------------------------------------------------
# 8. Trainer
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
# 9. Start LoRA Fine-Tuning
# --------------------------------------------------

print("\n======================================")
print("STARTING LoRA FINE-TUNING")
print("======================================\n")

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()


train_result = trainer.train()


# --------------------------------------------------
# 10. Training results
# --------------------------------------------------

print("\n======================================")
print("LoRA TRAINING COMPLETED")
print("======================================")

print("\nTraining metrics:")

print(
    train_result.metrics
)


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
# 11. Save LoRA adapter
# --------------------------------------------------

print("\nSaving LoRA model...")

trainer.save_model(
    "./bart_lora"
)

tokenizer.save_pretrained(
    "./bart_lora"
)

print("\nLoRA model saved to:")
print("./bart_lora")

print("\nLoRA fine-tuning finished successfully!")