# Fine-Tuning a Pre-trained Language Model for Automatic Text Paraphrasing

## 1. Project Overview

This project investigates the fine-tuning of a pre-trained sequence-to-sequence language model for automatic text paraphrasing.

The project compares three configurations:

1. BART-base (pre-trained baseline)
2. BART with Full Fine-Tuning (FFT)
3. BART with LoRA (Parameter-Efficient Fine-Tuning)

The objective is to investigate the effect of conventional full fine-tuning and parameter-efficient fine-tuning on paraphrase generation, while also comparing computational resource requirements.

---

## 2. Model

The selected pre-trained language model is:

**facebook/bart-base**

BART is a sequence-to-sequence Transformer model suitable for text generation tasks.

For the LoRA experiment, Low-Rank Adaptation (LoRA) was applied to the attention projection layers.

### LoRA configuration

- Rank (`r`): 8
- LoRA alpha: 16
- LoRA dropout: 0.1
- Target modules: `q_proj`, `v_proj`
- Trainable parameters: 442,368
- Total parameters: 139,862,784
- Trainable parameters: 0.3163%

---

## 3. Dataset

The project uses the English subset of the **PAWS-X** dataset.

Only examples labelled as paraphrases (`label = 1`) were used for the generative paraphrasing task.

### Dataset size

| Split | Examples |
|---|---:|
| Training | 21,829 |
| Validation | 863 |
| Test | 907 |

The input is `sentence1` and the target output is `sentence2`.

---

## 4. Experimental Methodology

Three experimental configurations were evaluated.

### 4.1 BART-base Baseline

The original pre-trained BART-base model was evaluated without task-specific fine-tuning.

### 4.2 Full Fine-Tuning

All BART model parameters were fine-tuned using the PAWS-X paraphrase training set.

Configuration:

- Epochs: 1
- Learning rate: `5e-5`
- Training batch size: 1
- Evaluation batch size: 1
- Gradient accumulation steps: 4
- Maximum sequence length: 128
- Mixed precision: FP16
- Random seed: 42

### 4.3 LoRA Fine-Tuning

LoRA was applied to BART-base while keeping the original model parameters frozen.

The same training dataset and main training settings were used to provide a comparable experiment.

---

## 5. Hardware

The experiments were performed on:

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU memory: 6 GB
- Operating system: Windows
- Python: 3.11.9
- PyTorch: 2.14.0 with CUDA support

---

## 6. Evaluation Metrics

The models were evaluated on the 907 paraphrase examples in the PAWS-X test set.

The following metrics were used:

- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore F1

Qualitative examples were also examined to assess how substantially the generated sentences differed from the original inputs.

---

## 7. Results

### Quantitative Results

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| BART-base | 0.6515 | 0.9314 | 0.7628 | 0.8485 | N/A |
| BART Full Fine-Tuning | 0.6480 | 0.9296 | 0.7678 | 0.8484 | N/A |
| BART LoRA | 0.6441 | 0.9274 | 0.7602 | 0.8466 | 0.9767 |

### Resource Comparison

| Model | Training Time | Peak GPU Memory | Trainable Parameters |
|---|---:|---:|---:|
| BART Full Fine-Tuning | 22.58 min | 2.59 GB | 139,862,784 |
| BART LoRA | 19.82 min | 0.88 GB | 442,368 |

LoRA trained approximately 0.3163% of the total BART parameters.

---

## 8. Qualitative Analysis

The generated outputs frequently remained close to the original input sentences.

For example, some generated outputs reproduced the original sentence with only minor grammatical, punctuation, or word-order changes.

This indicates that lexical-overlap metrics such as BLEU and ROUGE should be interpreted carefully for paraphrase generation. A generated sentence can obtain a high overlap score while making only a small change to the source sentence.

BERTScore was additionally used to provide a semantic similarity measure.

---

## 9. Key Findings

The experiments show that:

- The baseline BART model already produced high lexical-overlap scores on the selected test set.
- Full fine-tuning produced only modest changes in BLEU and ROUGE scores compared with the baseline.
- LoRA produced similar evaluation scores while updating only 0.3163% of the model parameters.
- LoRA required less peak GPU memory than full fine-tuning in this experiment.
- Qualitative examples indicate that substantial paraphrase generation remains challenging when evaluation relies primarily on reference-based similarity metrics.

---

## 10. Reproducibility

### Create the environment

```bash
python -m venv .venv