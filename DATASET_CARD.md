# Dataset Card — PAWS-X

## Dataset Name

PAWS-X (English subset)

## Source

The dataset is available through the Hugging Face Datasets library:

`google-research-datasets/paws-x`

Configuration used:

`en`

## Task

The project uses PAWS-X for automatic English text paraphrasing.

Each example contains two sentences and a label indicating whether the two sentences are paraphrases.

For this project, examples with:

`label = 1`

were selected as positive paraphrase pairs.

The task was then formulated as a sequence-to-sequence generation problem:

**Input sentence → Paraphrased sentence**

## Dataset Splits

The original English PAWS-X dataset used in the project contains:

| Split | Number of examples |
|---|---:|
| Training | 21,829 |
| Validation | 863 |
| Test | 907 |

The validation and test sets were retained for model evaluation.

## Data Preparation

The following preparation steps were applied:

1. The English (`en`) configuration of PAWS-X was loaded.
2. Only examples with `label = 1` were retained.
3. The sentence pairs were converted into input-target pairs for sequence-to-sequence training.
4. The training data was tokenized using the BART tokenizer.
5. Input and target sequences were limited to a maximum length of 128 tokens.

## Additional Training Data Experiment

An additional Full Fine-Tuning experiment investigated whether reducing highly similar sentence pairs could encourage more meaningful paraphrasing.

Training examples with a word-overlap ratio below `0.90` were selected.

This produced:

- 12,552 filtered training examples.
- Reverse sentence pairs were then created.
- The resulting training set contained 25,104 examples.

The validation and test sets were kept unchanged.

This experiment was used to investigate whether greater lexical variation in the training examples could reduce the model's tendency to copy the input sentence.

## Evaluation

The models were evaluated on the held-out test set containing 907 examples.

The following metrics were used:

- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore F1 for the LoRA experiment

These metrics were combined with qualitative inspection of generated paraphrases.

## Dataset Characteristics

PAWS-X contains sentence pairs designed to distinguish genuine paraphrases from sentences with high lexical overlap but different meanings.

This makes the dataset relevant to paraphrase modelling because lexical similarity alone does not guarantee semantic equivalence.

## Limitations

The dataset contains many sentence pairs with substantial lexical similarity.

As a result, a model trained directly on these examples may learn that copying or making only small changes to the input is sufficient to achieve strong lexical-overlap metrics.

This is relevant to the observed behaviour of the fine-tuned BART model, which frequently generated identical or near-identical sentences.

The dataset therefore provides useful quantitative evaluation but should be complemented with qualitative analysis when assessing actual paraphrase quality.

## Reproducibility

Dataset loading and preprocessing are implemented in the project source code.

Relevant files include:

- `code/prepare_data.py`
- `code/prepare_bart_data.py`
- `code/full_finetuning.py`
- `code/lora_finetuning.py`
- `code/check_overlap.py`

## Project Repository

`https://github.com/shubhamnagle22-glitch/LLM-Fine-Tuning-Paraphrasing`