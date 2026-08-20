# Gen-AI-Lab

A collection of 12 standalone Generative AI exercises using Hugging Face `transformers`, `diffusers`, and related libraries. Each script is self-contained and runnable on its own.

## Exercises

| File | Topic |
|---|---|
| `Genai1.py` | Text generation with GPT-2 (sampling vs. greedy decoding) |
| `Genai2.py` | Prompt engineering: zero-shot, few-shot, chain-of-thought |
| `Genai3.py` | Conversational chatbot using DialoGPT |
| `Genai4.py` | Text summarization (BART) and question answering (DistilBERT-SQuAD) |
| `Genai5.py` | Sentiment analysis and zero-shot document classification |
| `Genai6.py` | Retrieval-Augmented Generation (RAG) with FAISS + Flan-T5 |
| `Genai7.py` | Code generation and debugging with CodeGen |
| `Genai8.py` | Image generation with Stable Diffusion |
| `Geanai9.py` | Multimodal image captioning and VQA with BLIP |
| `Genai10.py` | Fine-tuning DistilBERT for text classification (IMDB) |
| `Genai11.py` | Integrated pipeline: text → image → audio content generation |
| `Genai12.py` | Deploying a summarizer with Gradio + ROUGE evaluation |

## Setup

```bash
pip install transformers diffusers torch sentence-transformers faiss-cpu \
    datasets scikit-learn gtts gradio evaluate rouge_score pillow requests
```

GPU is optional but recommended for `Genai8.py`, `Genai10.py`, and `Genai11.py`.

## Usage

Run any script directly:

```bash
python Genai1.py
```

Models are downloaded from the Hugging Face Hub on first run.
