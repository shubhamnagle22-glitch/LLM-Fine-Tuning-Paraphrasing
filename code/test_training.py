import torch
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "facebook/bart-base"

print("Loading dataset...")

dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only paraphrase pairs
dataset = dataset.filter(lambda x: x["label"] == 1)

# Small subset for the GPU test
train_data = dataset["train"].select(range(100))
validation_data = dataset["validation"].select(range(20))

print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))

print("Loading tokenizer and model...")

tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def tokenize_data(examples):
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

print("Tokenizing...")

train_data = train_data.map(
    tokenize_data,
    batched=True,
    remove_columns=train_data.column_names
)

validation_data = validation_data.map(
    tokenize_data,
    batched=True,
    remove_columns=validation_data.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = TrainingArguments(
    output_dir="./test_output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    fp16=True,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=validation_data,
    data_collator=data_collator,
)

print("\nStarting small GPU training test...")

trainer.train()

print("\nTraining test completed successfully!")

if torch.cuda.is_available():
    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
        "GB"
    )
    