from datasets import load_dataset

# Load PAWS-X English
dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

# Keep only actual paraphrase pairs
train_data = dataset["train"].filter(lambda x: x["label"] == 1)
validation_data = dataset["validation"].filter(lambda x: x["label"] == 1)
test_data = dataset["test"].filter(lambda x: x["label"] == 1)

print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Test examples:", len(test_data))

# Show one example
print("\nExample:")
print("Original:", train_data[0]["sentence1"])
print("Paraphrase:", train_data[0]["sentence2"])