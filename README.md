# Fake News Detection using BERT and FastAPI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![BERT](https://img.shields.io/badge/Model-BERT-red)

A deep learning project that detects fake news articles using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model. The project includes a model training pipeline and a real-time web application served via **FastAPI**.

## 📌 Overview

Misinformation spreads rapidly in the digital age. This project leverages **Transfer Learning** to distinguish between genuine and fake news with high accuracy. The system processes raw text, understands semantic context using BERT, and provides a probability score for the likelihood of the news being fake.

### Key Features
* **State-of-the-art NLP:** Uses `bert-base-uncased` for contextual feature extraction.
* **Robust Architecture:** Custom Keras layer wrapper for BERT, Dropout regularization, and Sigmoid classification.
* **Real-time Prediction:** Deployed using **FastAPI** for low-latency inference.
* **User Interface:** Simple HTML frontend to test news articles instantly.

## 🏗️ Neural Architecture

The model follows a 4-stage pipeline:
1.  **Tokenization:** Converts text into Input IDs and Attention Masks (Max length: 256).
2.  **BERT Encoder:** Extracts 768-dimensional contextual embeddings from the text.
3.  **Dropout Layer:** Applies a 20% dropout rate to prevent overfitting.
4.  **Classification Head:** A Dense layer with Sigmoid activation outputs a confidence score (0 = Real, 1 = Fake).

## 📂 Project Structure

```bash
├── news.ipynb                # Jupyter Notebook for Data Cleaning & Model Training
├── fake_news.py              # Main FastAPI application & Inference logic
├── bert_fakenews_model.keras # Saved trained model (generated after training)
├── templates/
│   └── index.html            # Frontend UI for the web app
├── README.md                 # Project documentation
└── requirements.txt          # List of dependencies

<img width="2240" height="1215" alt="Screenshot 2025-12-22 005050" src="https://github.com/user-attachments/assets/721848a4-126b-40bb-b9d1-7f21869d6c1c" />
<img width="2251" height="1211" alt="Screenshot 2025-12-22 005434" src="https://github.com/user-attachments/assets/321c69f3-e86b-4bc1-986a-417ed7c772bb" />

