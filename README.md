# BookRec - AI-Powered Book Recommendation System 📚✨

## Overview
BookRec is an AI-powered book recommendation system that leverages **collaborative filtering** (using NCF), **cosine similarity**, and **text-based embeddings** (via TextCNN) to suggest personalized book recommendations based on user preferences and book descriptions.

## Features
- **Hybrid Recommendation System**: Combines collaborative filtering, cosine similarity, and deep learning techniques.
- **User-Driven Interface**: Allows users to select a book and generate personalized recommendations.
- **Streamlit App**: Interactive, user-friendly, and web-based app for exploration.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/saidurpulok/BookRec.git
```

### 2. Navigate to the Project Directory
```bash
cd BookRec
```

### 3. Set Up the Environment
You need a Python virtual environment. Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset
The project uses the **Goodreads Books 100K Dataset**. You can download it from Kaggle using this [link](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/). Once downloaded, place the dataset files in the `Dataset/` directory.

---

## 🚀 Running the Application

### 1. Launch the Streamlit App
```bash
streamlit run app.py
```

This will start the application and give you a link to access it in your default web browser.

---

## 💡 Tech Stack
- **Python**
- **PyTorch** (for machine learning models)
- **Streamlit** (for the web-based dashboard)
- **Transformers** (for NLP embeddings via TextCNN)
- **NumPy, Pandas** (for data manipulation and handling)
- **Cosine Similarity**, **NCF Model**, and **TextCNN** for recommendation logic.

---

## 🧩 Models
### 1. **NCF (Neural Collaborative Filtering)**:
   - Collaborative filtering model using user and book IDs for prediction.
   - Trains user-item interaction embeddings to predict book preferences.

### 2. **TextCNN (Convolutional Neural Networks for Text)**:
   - Leverages descriptions of books and embeddings for similarity-based predictions.

### 3. **Cosine Similarity**:
   - Computes similarity scores between selected book features and other books in the database.

---

## 🔗 Links
- [Dataset - Goodreads Books 100K on Kaggle](https://www.kaggle.com/datasets/mdhamani/goodreads-books-100k/)
<!-- - [Streamlit Demo Link](#) -->

---

## 🏆 Contributions
This project is built using machine learning models and NLP techniques, primarily focusing on integrating hybrid approaches for better book recommendations.

---

### 🧑‍💻 Author
Saidur Rahman Pulok
[LinkedIn Profile](https://www.linkedin.com/in/mdsaidurrahmanpulok)  
[Personal Website](https://saidurpulok.github.io)

---

If you encounter any issue or would like to contribute, feel free to open an issue or pull request! 🚀