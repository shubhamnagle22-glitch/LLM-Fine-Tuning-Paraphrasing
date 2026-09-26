# Model Card — BART-base

## Model Name

BART-base (`facebook/bart-base`)

## Model Architecture

BART-base is an encoder-decoder Transformer language model. It consists of a bidirectional encoder and an autoregressive decoder.

The encoder processes the input sentence, while the decoder generates the output sequence. This encoder-decoder architecture is suitable for sequence-to-sequence generation tasks such as text paraphrasing.

## Model Size

- Model parameters: approximately 139 million
- Architecture: Encoder-Decoder Transformer
- Model family: BART
- Base model: `facebook/bart-base`

## Application

The model is used for automatic text paraphrasing.

The objective is to generate a sentence that preserves the meaning of the input while changing its wording or structure.

Example:

**Input:**  
He purchased a new laptop yesterday.

**Generated paraphrase:**  
He bought a new laptop yesterday.

## Fine-Tuning

The BART-base model was evaluated using three main approaches:

1. **Baseline:** The original pre-trained BART-base model was evaluated without fine-tuning.
2. **Full Fine-Tuning (FFT):** All model parameters were updated using the PAWS-X training data.
3. **LoRA:** Low-Rank Adaptation was applied to selected attention projection layers while keeping the majority of the original model parameters frozen.

An additional improved Full Fine-Tuning experiment was performed using lower-overlap paraphrase pairs and bidirectional training examples.

## Training Configuration

The main fine-tuning experiments used:

- Learning rate: `5e-5`
- Batch size: `1`
- Gradient accumulation steps: `4`
- Epochs: `1` for the main FFT and LoRA experiments
- Maximum sequence length: `128`
- Mixed precision: FP16
- Random seed: `42`

The LoRA configuration used:

- Rank (`r`): `8`
- LoRA alpha: `16`
- LoRA dropout: `0.1`
- Target modules: `q_proj`, `v_proj`
- Trainable parameters: `442,368`
- Trainable percentage: approximately `0.3163%`

## Hardware

The experiments were performed on a Windows 11 laptop with:

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU
- GPU memory: 6 GB

Full fine-tuning was successfully performed on the available GPU.

## Intended Use

This model adaptation is intended as an experimental academic project for automatic English sentence paraphrasing.

It can be used to demonstrate sequence-to-sequence language generation and compare conventional full fine-tuning with parameter-efficient fine-tuning.

## Limitations

The fine-tuned model does not consistently generate substantial paraphrases. In several test cases, it produces the original sentence or a near-copy with only minor changes.

This behaviour is reflected in the qualitative evaluation and demonstrates a limitation of the training data and fine-tuning approach.

The model should therefore not be assumed to produce semantically equivalent and stylistically diverse paraphrases for every input.

## Responsible Use

Generated text should be reviewed by a human before being used in academic, professional, or other important contexts.

The model may produce outputs that are copied, minimally changed, grammatically altered, or otherwise unsuitable as a true paraphrase.

## Software

The project uses:

- Python 3.11.9
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- Accelerate
- NLTK
- ROUGE
- BERTScore
- Streamlit

## Model Source

Hugging Face model:

`facebook/bart-base`

Project repository:

`https://github.com/shubhamnagle22-glitch/LLM-Fine-Tuning-Paraphrasing`