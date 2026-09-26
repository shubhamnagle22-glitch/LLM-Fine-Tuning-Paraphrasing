from datasets import load_dataset
import re

print("Loading PAWS-X...")

dataset = load_dataset(
    "google-research-datasets/paws-x",
    "en"
)

dataset = dataset["train"].filter(lambda x: x["label"] == 1)

print("Total paraphrase pairs:", len(dataset))


def word_overlap(sentence1, sentence2):
    words1 = set(re.findall(r"\b\w+\b", sentence1.lower()))
    words2 = set(re.findall(r"\b\w+\b", sentence2.lower()))

    if not words1 or not words2:
        return 1.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    return len(intersection) / len(union)


thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]

print("\nTraining examples remaining at each threshold:")

for threshold in thresholds:

    count = sum(
        word_overlap(x["sentence1"], x["sentence2"]) < threshold
        for x in dataset
    )

    print(
        f"Overlap < {threshold:.2f}: {count} examples"
    )