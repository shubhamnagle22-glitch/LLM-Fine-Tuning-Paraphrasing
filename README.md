# Fine-Tuning a Pre-trained Language Model for Automatic Text Paraphrasing

## 1. Project Overview

This project investigates the fine-tuning of a pre-trained sequence-to-sequence language model for automatic English text paraphrasing.

The project compares:

1. BART-base (pre-trained baseline)
2. BART with Full Fine-Tuning (FFT)
3. BART with LoRA (Parameter-Efficient Fine-Tuning)
4. Improved BART Full Fine-Tuning using data filtering and bidirectional augmentation

The objective is to investigate the effect of conventional full fine-tuning and parameter-efficient fine-tuning on paraphrase generation, while comparing computational resource requirements and qualitative generation behaviour.

---

## 2. Model and Architecture

The selected pre-trained language model is:

**facebook/bart-base**

BART is an encoder-decoder Transformer model designed for sequence-to-sequence generation.

### Architecture Justification

The selected application is automatic text paraphrasing, where an input sentence must be transformed into an output sentence while preserving its meaning.

BART uses an encoder-decoder architecture:

- The encoder processes and represents the input sentence.
- The decoder generates the target output sequence.

This architecture is appropriate for the paraphrasing task because the problem can be formulated as:

**Input sentence → Paraphrased sentence**

For example:

**Input:**

> He purchased a new laptop yesterday.

**Generated paraphrase:**

> He bought a new laptop yesterday.

Further model information is documented in:

`MODEL_CARD.md`

---

## 3. Model Configuration

### BART-base

- Model: `facebook/bart-base`
- Architecture: Encoder-Decoder Transformer
- Total parameters: 139,862,784
- Task: Sequence-to-sequence text generation

### LoRA Configuration

For the LoRA experiment, Low-Rank Adaptation was applied to selected attention projection layers while keeping the original model parameters frozen.

- Rank (`r`): 8
- LoRA alpha: 16
- LoRA dropout: 0.1
- Target modules: `q_proj`, `v_proj`
- Trainable parameters: 442,368
- Total parameters: 139,862,784
- Trainable percentage: 0.3163%

---

## 4. Dataset

The project uses the English subset of the **PAWS-X** dataset.

Dataset configuration:

`google-research-datasets/paws-x`

Configuration:

`en`

Only examples labelled as paraphrases (`label = 1`) were used for the generative paraphrasing task.

The input is `sentence1` and the target output is `sentence2`.

The dataset preparation details are documented in:

`DATASET_CARD.md`

### Original Dataset Size

| Split | Examples |
|---|---:|
| Training | 21,829 |
| Validation | 863 |
| Test | 907 |

### Improved Training Dataset

An additional Full Fine-Tuning experiment was conducted to investigate whether reducing highly similar training pairs would encourage more meaningful paraphrasing.

A word-overlap measure was calculated between `sentence1` and `sentence2`.

Training pairs with word overlap below `0.90` were retained.

This reduced the training set from:

**21,829 → 12,552 pairs**

The retained training pairs were then augmented bidirectionally:

- `sentence1 → sentence2`
- `sentence2 → sentence1`

This resulted in:

**25,104 training examples**

The validation and test sets were not modified.

---

## 5. Experimental Methodology

### 5.1 BART-base Baseline

The original pre-trained BART-base model was evaluated without task-specific fine-tuning.

This provides a baseline against which the fine-tuned models can be compared.

### 5.2 Full Fine-Tuning

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

### 5.3 LoRA Fine-Tuning

LoRA was applied to BART-base while keeping the original model parameters frozen.

The main training settings were kept comparable with the Full Fine-Tuning experiment.

LoRA updated only 442,368 parameters, representing 0.3163% of the total model parameters.

### 5.4 Improved Full Fine-Tuning

A second Full Fine-Tuning experiment was performed using the filtered and bidirectionally augmented training data.

Configuration:

- Training examples: 25,104
- Validation examples: 863
- Test examples: 907
- Epochs: 1
- Learning rate: `5e-5`
- Training batch size: 1
- Evaluation batch size: 1
- Gradient accumulation steps: 4
- Maximum sequence length: 128
- Mixed precision: FP16
- Random seed: 42

The improved model was saved as:

`bart_full_finetuned_improved`

---

## 6. Hardware

The experiments were performed on:

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU memory: 6 GB
- Operating system: Windows 11
- Python: 3.11.9
- PyTorch: 2.14.0 with CUDA support
- Transformers: 5.17.0
- Datasets: 5.0.1
- PEFT: 0.21.0

Full Fine-Tuning was successfully performed on the available 6 GB GPU.

---

## 7. Software and Libraries

The project was implemented using Python and the following libraries:

- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Accelerate
- NLTK
- `rouge_score`
- BERTScore
- Streamlit

The complete Python dependencies are provided in:

`requirements.txt`

---

## 8. Evaluation Metrics

The models were evaluated on the 907 paraphrase examples in the PAWS-X test set.

The following metrics were used:

- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore F1

Qualitative examples were also examined to assess how substantially the generated sentences differed from the original inputs.

---

## 9. Results

### 9.1 Quantitative Results

| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---:|---:|---:|---:|---:|
| BART-base | 0.6515 | 0.9314 | 0.7624 | 0.8486 | N/A |
| BART Full Fine-Tuning | 0.6480 | 0.9295 | 0.7678 | 0.8486 | N/A |
| BART LoRA | 0.6433 | 0.9271 | 0.7613 | 0.8471 | 0.9766 |
| BART Improved Full Fine-Tuning | 0.6480 | 0.9280 | 0.7687 | 0.8481 | N/A |

### 9.2 Resource Comparison

| Model | Training Time | Peak GPU Memory | Trainable Parameters | Trainable % |
|---|---:|---:|---:|---:|
| BART Full Fine-Tuning | 29.88 min | 2.59 GB | 139,862,784 | 100% |
| BART LoRA | 20.88 min | 0.88 GB | 442,368 | 0.3163% |
| BART Improved Full Fine-Tuning | 90.75 min | 2.58 GB | 139,862,784 | 100% |

The improved Full Fine-Tuning experiment required more training time because it used 25,104 bidirectional training examples compared with 21,829 examples in the original Full Fine-Tuning experiment.

---

## 10. Qualitative Analysis

The generated outputs frequently remained close to the original input sentences.

For example, the fine-tuned model sometimes reproduced the input almost exactly:

**Input:**

> She purchased a new laptop yesterday.

**Generated:**

> She purchased a new laptop yesterday.

However, the improved model was also capable of producing meaningful lexical changes in some cases.

**Input:**

> He purchased a new laptop yesterday.

**Generated:**

> He bought a new laptop yesterday.

This represents a meaningful lexical paraphrase because the wording changed while preserving the original meaning.

Other examples showed smaller transformations, such as word-order changes or grammatical corrections.

The qualitative results demonstrate that reference-based metrics should be interpreted carefully for paraphrase generation. A generated sentence can obtain a high BLEU or ROUGE score while remaining very similar to the source sentence.

---

## 11. Key Findings

The experiments show that:

- The baseline BART model already produced high lexical-overlap scores on the selected test set.
- Full Fine-Tuning produced only modest changes in BLEU and ROUGE compared with the baseline.
- LoRA achieved similar evaluation scores while updating only 0.3163% of the model parameters.
- LoRA used substantially less peak GPU memory than Full Fine-Tuning.
- Filtering highly similar training pairs and adding bidirectional training did not produce a substantial improvement in the automatic evaluation scores.
- The improved Full Fine-Tuning model produced some meaningful lexical paraphrases but still frequently generated outputs close to the original input.
- Qualitative analysis is therefore important when evaluating automatic paraphrasing systems.

---

## 12. Streamlit Demonstration

A Streamlit interface was developed to demonstrate the fine-tuned model interactively.

The application allows a user to:

1. Enter an input sentence.
2. Generate a paraphrase using the fine-tuned BART model.
3. View the generated output.

The current demonstration uses:

`bart_full_finetuned_improved`

Run the application using:

```bash
python -m streamlit run app.py