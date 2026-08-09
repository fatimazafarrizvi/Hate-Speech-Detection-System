<div align="center">

<img src="assets/banner.png" alt="Hate Speech Detection using XLM-RoBERTa" width="100%"/>

# 🛡️ Hate Speech Detection System

**A transformer-based hate speech classifier with a real-time Chrome extension for Instagram content moderation.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.53.3-yellow)](https://huggingface.co/docs/transformers)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension%20MV3-4285F4?logo=googlechrome&logoColor=white)](https://developer.chrome.com/docs/extensions/mv3/intro/)

</div>

---

## 👥 Team

Developed as a final-year project at **Techno Main Salt Lake**.

| Name | Role |
|---|---|
|Fatima Zafar Rizvi | Research, RNN and XLM-RoBERTa Model Training |
| MD Saad Alam | Research, XLM-RoBERTa Model and Chrome Extension Development |
| Harsh Sharma | Chrome Extension Development and Documentation |
| Sneha Singh | Documentation and Presentation |

## 📖 Overview

This project fine-tunes **XLM-RoBERTa**, a multilingual transformer model, to classify text as **Hate** or **Non-Hate** speech, and ships that model as a working end-to-end product rather than just a notebook.

<div align="center">

# 🛡️ XLM-Roberta Architecture

<img src="asset1/XLM Roberta Model architecture.png" alt="XLM-Roberta_architecture" width="90%"/>

</div>

A local **FastAPI** server loads the trained model and exposes a prediction endpoint, and a **Chrome extension (Manifest V3)** talks to that server to:

- Scan Instagram posts, captions, and comments live as you scroll
- Highlight hate speech in red with a confidence badge (e.g. `Hate 98%`)
- Let you analyze any text through a popup or right-click context menu

The goal is real-time, on-device-adjacent content moderation — the extension never sends your data anywhere outside your own local API.

## ✨ Key Features

- 🌍 **Multilingual detection** — built on XLM-RoBERTa, which supports 100 languages
- ⚡ **Real-time scanning** — automatically detects and highlights hate speech on Instagram as you browse
- 🧩 **Chrome Extension (Manifest V3)** — popup analyzer, right-click "Analyze for hate speech," and a floating on-page panel with Pause/Resume and Rescan controls
- 🔌 **REST API (FastAPI)** — `/predict` and `/predict/batch` endpoints, plus a `/health` check
- 🎚️ **Adjustable confidence threshold** — tune sensitivity from the extension settings (default 55%)
- 🔒 **Privacy-first** — the model runs locally; no text is sent to any third-party service
- 📊 **Evaluated performance** — precision/recall, ROC, confusion matrix, and overfitting analysis included in `assets/`

## 🖼️ Screenshots

<table>
<tr>
<td width="50%">
<img src="assets/screenshots/instagram_demo_1.png" alt="Live Instagram comment highlighting"/>
<p align="center"><em>Live highlighting on Instagram comments, with a running scan summary</em></p>
</td>
<td width="50%">
<img src="assets/screenshots/instagram_demo_2.png" alt="Live Instagram highlighting, second example"/>
<p align="center"><em>Hate captions/comments flagged in red with a confidence badge</em></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="assets/screenshots/prediction_hate.png" alt="CLI prediction — hate example"/>
<p align="center"><em>CLI inference: high-confidence hate prediction</em></p>
</td>
<td width="50%">
<img src="assets/screenshots/prediction_non_hate.png" alt="CLI prediction — non-hate example"/>
<p align="center"><em>CLI inference: non-hate prediction</em></p>
</td>
</tr>
</table>

## 🏗️ Architecture

```
Chrome Extension  --->  FastAPI (localhost:8000)  --->  XLM-RoBERTa model
     │                          │
  popup / context menu      POST /predict
  Instagram content script  GET  /health
```

The model is too large to run inside the browser, so the extension talks to a local Python server that loads it once at startup and serves predictions over HTTP.

```
Hate-Speech-Detection-System/
├── api/                        # FastAPI server + inference logic
│   ├── server.py               #   REST endpoints: /health, /predict, /predict/batch
│   └── inference.py            #   Model loading & prediction (HateSpeechClassifier)
├── Chrome Extension/            # Manifest V3 browser extension
│   ├── manifest.json
│   ├── popup.html / .js / .css # Popup analyzer UI
│   ├── background.js           # Context menu + API calls
│   ├── content.js / instagram.js / content.css  # Instagram live scanning
│   ├── options.html / .js      # API URL & settings
│   └── icons/
├── model/                      # CLI tooling & extension setup docs
│   ├── predict.py              # Standalone CLI prediction script
│   ├── CHROME_EXTENSION_SETUP.md
│   └── requirements.txt
├── notebooks/                  # Training & data-prep notebooks
│   ├── A_New_Exp.ipynb         # XLM-RoBERTa fine-tuning
│   └── Data_Extraction.ipynb
├── rnn_hate_speech_model.ipynb # Baseline RNN model (for comparison)
├── datasets/                   # Cleaned & merged training data (text, label)
├── assets/ & asset/            # Banner, screenshots, training curves
├── docs/                       # PRD and supporting documentation
└── scripts/                    # Icon generation utilities
```

<div align="center">

# 🛡️ ML Architecture of Project

<img src="asset1/ML architecture fy proj.png" alt="ML_architecture" width="90%"/>

</div>

## 🧠 Model & Data

- **Base model:** `xlm-roberta-base` fine-tuned for binary sequence classification (`Non-Hate` / `Hate`)
- **Dataset:** an English hate-speech dataset of **44,207 labeled samples**; URLs, HTML tags, duplicates, and extra whitespace were stripped during cleaning, while emojis, hashtags, and punctuation were **preserved** to retain emotional/contextual signal
- **Tokenization:** XLM-RoBERTa tokenizer (byte-level BPE), input IDs + attention mask, max sequence length 128–256
- **Data split:** 80% train / 10% validation / 10% test, stratified by label
- **Architecture:** 12-layer transformer encoder, hidden size 768, 12 attention heads, dropout 0.1, classification head (768 → 2) on the `<s>` token representation
- **Training config:** 5 epochs, batch size 16 (gradient accumulation ×2), learning rate `2e-5`, weight decay 0.01, warmup ratio 0.10, mixed precision (fp16), gradient checkpointing, `AdamW` optimizer, cross-entropy loss, early stopping on F1
- A separate **RNN baseline** (`rnn_hate_speech_model.ipynb`) was built first and then outperformed by the transformer approach (see comparison below)

## 📈 Performance

| Metric | RNN Baseline | XLM-RoBERTa (final) | Improvement |
|---|---|---|---|
| Accuracy | 89.70% | **92.36%** | +2.6% |
| Precision | 86% | **87.9%** | +1.9% |
| Recall | 86% | **90.95%** | +4.95% |
| F1-Score | 86% | **89.4%** | +3.4% |
| ROC-AUC | — | **0.9731** | — |

<div align="center">

# 🛡️ From RNN to XLM-Roberta 

<img src="asset1/rnn to xlm.png" alt="rnn to xlm" width="90%"/>

</div>

The RNN struggled with long-range dependencies, sequential processing speed, and multilingual/contextual nuance — which is why the project moved to a transformer-based architecture (`FROM RNN TO XLM-RoBERTa` in `docs/`). The final XLM-RoBERTa model shows higher precision, recall, and F1, better generalization, and minimal overfitting.


# 📊 XLM-Roberta Model Valuation Graphs

<table>
<tr>
<td width="50%"><img src="assets/Confusion_matrix.png" alt="Confusion Matrix"/></td>
<td width="50%"><img src="assets/ROC curve.png" alt="ROC Curve"/></td>
</tr>
<tr>
<td width="50%"><img src="assets/Training vs Validation Loss Curve.png" alt="Training vs Validation Loss"/></td>
<td width="50%"><img src="assets/Precision-Recall Curve.png" alt="Precision-Recall Curve"/></td>
</tr>
</table>

- **Confusion matrix:** 6,900 true negatives, 3,707 true positives, 508 false positives, 369 false negatives
- **ROC curve:** AUC of 0.9731, indicating strong separation between hate and non-hate classes
- **Training vs. validation loss:** both curves decrease steadily and stay closely aligned, showing stable convergence with minimal overfitting
- **Validation accuracy:** climbs from ~81.9% to ~92.4% across training steps
- **Prediction confidence:** the large majority of predictions cluster near 1.0 confidence, indicating the model is rarely "unsure"

Additional curves (F1 score, overfitting analysis, confidence histogram) are available in `assets/`.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Google Chrome
- ~2 GB free disk space (model + dependencies)
- The fine-tuned model folder `HateSpeech_XLMRoBERTa_Final/` (not included in this repo due to size — see [Model Weights](#-model-weights))

### 1. Install dependencies

```bash
git clone https://github.com/fatimazafarrizvi/Hate-Speech-Detection-System.git
cd Hate-Speech-Detection-System
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r model/requirements.txt
```

### 2. Generate extension icons

```bash
python scripts/generate_icons.py
```

### 3. Start the API server

```bash
python -m api.server
```

Wait for:

```
Loading hate speech model...
Model ready.
```

The server runs at `http://127.0.0.1:8000`. Verify it:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

### 4. Load the Chrome extension

1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked** and select the `Chrome Extension/` folder
4. Pin the extension from the toolbar

### 5. Use it

- **Instagram live scan:** open instagram.com while the API server is running — hateful comments/captions get highlighted in red with a confidence badge
- **Popup analysis:** click the extension icon, paste text, click Analyze
- **Right-click:** select text on any page → **Analyze for hate speech**

Full setup, troubleshooting, and environment variable reference: [`model/CHROME_EXTENSION_SETUP.md`](model/CHROME_EXTENSION_SETUP.md)

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server + model load status |
| `/predict` | POST | Classify a single text (`{"text": "..."}`) |
| `/predict/batch` | POST | Classify up to 32 texts at once (`{"texts": [...]}`) |

**Example response:**

```json
{
  "prediction": 1,
  "label": "Hate",
  "confidence": 0.9123,
  "probabilities": { "non_hate": 0.0877, "hate": 0.9123 }
}
```

| Environment Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./HateSpeech_XLMRoBERTa_Final` | Path to the fine-tuned model folder |
| `API_HOST` | `127.0.0.1` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |

## 🏷️ Model Weights

The fine-tuned `HateSpeech_XLMRoBERTa_Final/` checkpoint isn't checked into this repo (it's ~1 GB). Train it yourself with `notebooks/A_New_Exp.ipynb`, or host the weights externally (e.g. Hugging Face Hub / Google Drive) and point `MODEL_PATH` at the downloaded folder.

## 🛠️ Tech Stack

**Language:** Python 3.12
**ML/NLP:** PyTorch, Hugging Face Transformers, XLM-RoBERTa, scikit-learn
**Data:** Pandas, NumPy
**Visualization:** Matplotlib, Seaborn
**Backend:** FastAPI, Uvicorn, Pydantic
**Extension:** Chrome Manifest V3, vanilla JS/HTML/CSS
**Training environment:** Google Colab & Kaggle

## 💡 Applications

- Social media content moderation
- Online forums and community platforms
- News website comment filtering
- Educational discussion portals
- Customer support message screening
- Government and law enforcement content monitoring

## ⚠️ Limitations

- Currently supports **English text only**
- Performance depends on the quality and balance of the training dataset
- Computationally intensive due to the transformer architecture; benefits from a GPU for training/fine-tuning
- May struggle with sarcasm, slang, or highly implicit hate speech

## 🗺️ Future Scope

- [ ] Multilingual hate speech detection
- [ ] Cloud-based deployment for scalability (remove the localhost dependency)
- [ ] Extend live scanning beyond Instagram to other platforms
- [ ] Publish to the Chrome Web Store
- [ ] Explainable AI integration (e.g. SHAP, LIME) for prediction transparency
- [ ] Continuous model improvement with new datasets

## 📄 License

No license has been specified for this project yet. If you intend for others to reuse this code, consider adding a `LICENSE` file (e.g. MIT, Apache-2.0).

## 🙏 Acknowledgements

Built on [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) by Facebook AI, via the Hugging Face [Transformers](https://github.com/huggingface/transformers) library.

## 📚 References

1. A Comprehensive Survey on Multilingual Hate Speech Detection and Counterspeech Generation — [arXiv:2603.19279](https://arxiv.org/abs/2603.19279)
2. Comparative Analysis of Machine Learning, Transformer Models and Large Language Models for Hate Speech Detection — [arXiv:2603.04698](https://arxiv.org/abs/2603.04698)
3. Explainable Hate Speech Detection, ACM — [DOI:10.1145/3820759](https://dl.acm.org/doi/10.1145/3820759)
4. Recent Advances in Hate Speech Detection using Transformer-based Models — [arXiv:2508.04913](https://arxiv.org/abs/2508.04913)
5. Hate Speech Detection: Challenges, Methods and Future Directions, Social Network Analysis and Mining (Springer) — [DOI:10.1007/s13278-024-01361-3](https://link.springer.com/article/10.1007/s13278-024-01361-3)
6. Hate Speech Detection using Natural Language Processing and Deep Learning: A Systematic Review, Journal of Systems Architecture, Elsevier — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231223003557)
7. Explainable Hate Speech Detection: A Comprehensive Review, Expert Systems, Wiley — [DOI:10.1111/exsy.13562](https://onlinelibrary.wiley.com/doi/10.1111/exsy.13562)
8. Fortuna, P., & Nunes, S., A Survey on Automatic Hate Speech Detection in Text, *Information*, 13(6), 273, MDPI, 2022 — [MDPI](https://www.mdpi.com/2078-2489/13/6/273)
9. Poletto, F., Basile, V., Sanguinetti, M., Bosco, C., & Patti, V., Resources and Benchmark Corpora for Hate Speech Detection: A Systematic Review, *PeerJ Computer Science*, 7:e598, 2021
