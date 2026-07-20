# Hate Speech Detection System

A comprehensive machine learning solution for detecting and classifying hate speech in social media comments using advanced NLP models and real-time deployment capabilities.

## 📋 Overview

This final year project develops a multi-model hate speech detection system with real-time deployment via a web application and Chrome extension. The system progresses from an RNN baseline model to an optimized XLM-RoBERTa transformer model, achieving **AUC = 0.95** and deployment on Instagram comment sections through our Chrome extension.

## 🛠️ Technologies Used

### Machine Learning & NLP
- **Python** - Core programming language
- **TensorFlow/Keras** - Deep learning framework for RNN model
- **Transformers (Hugging Face)** - XLM-RoBERTa pre-trained model
- **Scikit-learn** - Model evaluation and metrics

### Data & Database
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Vector Database** - For efficient similarity search and embedding storage

### Deployment & Frontend
- **Flask/FastAPI** - Backend API
- **React/JavaScript** - Web application frontend
- **Chrome Extension API** - Instagram comment highlighting
- **Vercel** - Production web hosting

### Evaluation Tools
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Confusion matrix, ROC curve, precision-recall metrics

## 🎯 Model Development Progression

### Phase 1: RNN Model
We started with a Recurrent Neural Network baseline to establish initial performance metrics:

#### RNN Confusion Matrix
![RNN Confusion Matrix](asset/rnn%20confiusion%20matrix.jpg)

#### RNN Model Output Distributions
- **Hate Speech Distribution**
  ![Hate RNN](asset/hate_rnn.jpg)

- **Offensive Speech Distribution**
  ![Offensive RNN](asset/offensive_rnn.jpg)

- **Neither Category Distribution**
  ![Neither RNN](asset/neither_rnn.jpg)

### Phase 2: XLM-RoBERTa Fine-tuning
We then fine-tuned the XLM-RoBERTa multilingual transformer model, which significantly improved accuracy and provided better generalization across languages.

#### XLM-RoBERTa Architecture
![XLM-RoBERTa Architecture](asset/XLM-Roberta_architecture.jpg)

## 📊 Model Performance & Evaluation Metrics

### Confusion Matrix
The final model demonstrates excellent classification performance across all categories:

![Confusion Matrix](asset/confusion_matrix.jpg)

### Accuracy Curve
Training and validation accuracy progression showing model convergence:

![Accuracy Curve](asset/Accuracy_Curve.jpg)

### Loss Curve
Training and validation loss demonstrating model stability:

![Loss Curve](asset/Loss_Curve.jpg)

### ROC Curve
Receiver Operating Characteristic curve showing **AUC = 0.95**:

![ROC Curve](asset/ROC_Curve.jpg)

## 💻 Project Components

### Jupyter Notebooks
1. **rnn_hate_speech_model.ipynb** - RNN baseline model development and training
2. **A_New_Exp.ipynb** - XLM-RoBERTa fine-tuning and advanced experiments

### Chrome Extension
Complete browser extension for real-time detection on Instagram comment sections. Highlights hate and non-hate speech with visual indicators.

**[Chrome Extension Repository](https://github.com/Saad-tech1606/Hate-Speech-Detection)** - Access all chrome extension related files, implementation details, and deployment instructions.

## 📁 Data & Resources

### Dataset
- **[Database Link](https://drive.google.com/drive/folders/1JYIjLDJ1e4aW6uBEA3n111VOKdozelj1?usp=sharing)** - Complete training and evaluation datasets

### Data Cleaning Process
- **[Kaggle Data Cleaning Notebook](https://www.kaggle.com/code/fatimarizvi2/data-cleaning)** - Detailed data preprocessing and feature engineering pipeline

### Live Deployment
- **[Web Application](https://safespeak-sepia.vercel.app/)** - Live demo of the hate speech detection system

## 👥 Contributors

This project was developed by a team of 4 contributors:

- **Fatima Zafarrizvi** - Lead Developer & ML Engineer
- **Saad Ahmed** - Chrome Extension & Frontend Development
- **Team Members** - Data annotation, testing, and documentation

## 🚀 Features

✅ **Multi-model Comparison** - RNN vs Transformer models  
✅ **Multilingual Support** - XLM-RoBERTa handles multiple languages  
✅ **Real-time Detection** - Chrome extension for live comment analysis  
✅ **High Accuracy** - AUC score of 0.95  
✅ **Web Application** - User-friendly interface for text classification  
✅ **Vector DB Integration** - Efficient similarity search capabilities  
✅ **Comprehensive Evaluation** - Confusion matrix, ROC, precision-recall metrics

## 📈 Performance Summary

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.95 |
| **Model Type** | XLM-RoBERTa (Fine-tuned) |
| **Real-time Deployment** | ✓ Chrome Extension |
| **Web Application** | ✓ Live on Vercel |
| **Multi-language Support** | ✓ Yes |

## 📖 Usage

1. **Local Testing**
   - Run the Jupyter notebooks to train and evaluate models
   - Use the web application for text classification

2. **Chrome Extension**
   - Install from the [Chrome Extension Repository](https://github.com/Saad-tech1606/Hate-Speech-Detection)
   - Automatically highlights hate/non-hate speech on Instagram

3. **Online Demo**
   - Visit [SafeSpeak](https://safespeak-sepia.vercel.app/) for instant classification

## 🔗 Project Links

- 🌐 **[Live Web App](https://safespeak-sepia.vercel.app/)**
- 📊 **[Dataset](https://drive.google.com/drive/folders/1JYIjLDJ1e4aW6uBEA3n111VOKdozelj1?usp=sharing)**
- 🔧 **[Chrome Extension Code](https://github.com/Saad-tech1606/Hate-Speech-Detection)**
- 🧹 **[Data Cleaning Notebook](https://www.kaggle.com/code/fatimarizvi2/data-cleaning)**

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**Last Updated:** July 2026  
**Status:** ✅ Deployed and Active
