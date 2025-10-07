# LegalMind AI — Justice Meets Intelligence

### *An Intelligent Legal Reasoning System for Pakistani Law*

LegalMind AI is an advanced **Retrieval-Augmented Generation (RAG)** and **LegalBERT-powered** platform that brings together **law book retrieval**, **judgment summarization**, and **verdict prediction** — enabling deep insights and reasoning in Pakistani criminal law.

---

## Features

###  Law Book Retrieval (RAG)

* Digitizes and semantically indexes **CrPC 1898**, **PPC**, and **Qanun-e-Shahadat 1984**
* Uses **InLegalBERT** embeddings for domain-specific legal semantics
* Retrieves and explains relevant sections based on natural-language queries

###  Judgment Summarization

* Extracts judgments from High Court and Supreme Court archives
* Identifies **case type**, **court**, **arguments**, **laws invoked**, and **final decision**
* Generates structured, human-readable summaries using **GPT-based LLMs**

###  Verdict Prediction

* Trained **LegalBERT classifier** predicts likely verdicts (e.g., *Guilty*, *Dismissed*, *Granted*)
* Takes structured case summaries and arguments as input
* Supports end-to-end pipeline: `Raw Text → Structured Fields → Predicted Verdict`

###  Multi-Agent Legal Assistant

* Integrates all functionalities via **LangChain Agents** and **Tools**
* Agents can:

  * Retrieve statutory law (`lawbook_tool`)
  * Retrieve relevant case law (`judgment_tool`)
  * Predict likely verdicts (`predict_verdict_from_text`)
* Supports **memory-based conversations** using LangChain's `ConversationBufferMemory`

###  Gradio Frontend

* Interactive UI with 4 functional tabs:

  *  *Predict Verdict*
  *  *Search Law Books*
  *  *Past Case Judgments*
  *  *Legal ChatBot*
* Built using **Gradio Blocks** for real-time interaction

---

##  Core Architecture

| Component            | Description                                                 |
| -------------------- | ----------------------------------------------------------- |
| **Embeddings**       | `InLegalBERT` (custom mean-pooling on `law-ai/InLegalBERT`) |
| **Vector Store**     | `FAISS` (for both law books & judgments)                    |
| **LLM Backend**      | `OpenAI GPT-3.5-Turbo` via `langchain-openai`               |
| **Classifier Model** | Fine-tuned `nlpaueb/legal-bert-base-uncased`                |
| **Frameworks**       | LangChain, HuggingFace Transformers, Gradio                 |
| **Storage**          | Vector indexes & model checkpoints on Google Drive          |
| **Environment**      | Google Colab (GPU recommended)                              |

---

## Directory Overview

```
LegalMind/
│
├── lawbook_chunks.json          # Preprocessed law book text chunks
├── judgments_chunks.json        # Extracted court judgments
├── lawbook_store/               # FAISS index for law books
├── judgment_store/              # FAISS index for judgments
├── best_epoch18_model/          # Fine-tuned LegalBERT model
├── LegalMind.json               # Dataset used for training verdict predictor
└── app.py                       # Full RAG + Agent + Gradio pipeline
```

---

## Installation

```bash
!pip install langchain faiss-cpu transformers sentence-transformers pymupdf gradio
!pip install -U langchain-community langchain-openai
```

Mount Google Drive (if on Colab):

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Usage

### Step 1 — Index Law Books & Judgments

```python
lawbook_chunks = load_and_split_law_books()
judgment_chunks = split_judgments_by_common_headers("/content/drive/MyDrive/LegalMind/Judgments.pdf")
create_vectorstores(judgment_chunks, lawbook_chunks)
```

### Step 2 — Run RAG Pipeline

```python
response = main_chain_law.invoke("Explain Section 489-F of the Pakistan Penal Code.")
print(response)
```

### Step 3 — Predict Verdict

```python
case_text = "Petitioner was charged under Section 489-F for cheque dishonor..."
run_pipeline(case_text)
```

### Step 4 — Launch Gradio App

```python
demo.launch(share=True)
```

---

## Model Training

Fine-tuned **LegalBERT** for multi-class verdict classification:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=num_labels
)
```

Metrics tracked:

* Accuracy
* Precision / Recall / F1
* Confusion Matrix Visualization

---

## Results Snapshot

| Metric                        | Value                                    |
| ----------------------------- | ---------------------------------------- |
| **Final Validation Accuracy** | 97.4%                                    |
| **Macro F1 Score**            | 0.96                                     |
| **Top Labels**                | Dismissed, Granted, Convicted, Acquitted |
| **Inference Time**            | ~0.5s per case on T4 GPU                 |

---

## Agents Summary

| Tool                        | Function                                                  |
| --------------------------- | --------------------------------------------------------- |
| `lawbook_tool`              | Retrieves and explains relevant legal sections            |
| `judgment_tool`             | Fetches top court judgments with structured summaries     |
| `predict_verdict_from_text` | Predicts the likely outcome based on arguments & sections |

---

## Example Query Flow

**User:** “What does Section 489-F say about cheque dishonor?”
**Agent:** Retrieves PPC section → Explains the punishment and ingredients of the offence.

**User:** “Has any court ruled on this before?”
**Agent:** Summarizes top relevant judgments.

**User:** “Predict the likely verdict for this case.”
**Agent:** Extracts structured data → Predicts outcome using LegalBERT.

---

## Future Enhancements

*  Add Supreme Court case indexing
*  Support for PECA (Cybercrime Law) and NAB Ordinance
*  Integration with RAG-Fusion for multi-source reasoning
*  Persona-based Legal Chat (Lawyer / Judge / Student modes)
*  Retrieval performance benchmarking

---

##  Author

**CH Abdullah**
 *[contact.abdullah.ai@gmail.com](mailto:contact.abdullah.ai@gmail.com)*
 [LinkedIn](https://linkedin.com/in/) | [GitHub](https://github.com/)

