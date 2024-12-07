import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import pandas as pd
import papermill as pm
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from Models.models import NCF, TextCNN

max_words = 5000  # Vocabulary size
max_len = 200  # Maximum sequence length
num_users = 1000
num_items = 25151

# Set the app title, favicon, and layout
st.set_page_config(page_title='BookRec: An AI Book Recommender', page_icon='📚', layout='wide')

# Apply custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            background-color: #f0f2f6;
            color: #333;
        }
        .stApp {
            background-color: #010118;
            padding: 20px;
        }
        .title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: 700;
            color: #cfcfcf;
            margin-bottom: 30px;
        }
        .subtitle {
            text-align: center;
            color: #dfdfdf;
        }
        .book-card {
            padding: 15px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 350px;
            width: 250px;
        }
        .book-card img {
            border-radius: 5px;
            margin-bottom: 10px;
            width: 150px;
            height: 200px;
            object-fit: cover;
        }
        .book-card p {
            margin: 5px 0;
            color: #555;
            font-size: 0.9rem;
        }
        .book-card .title {
            font-weight: 600;
            color: #222;
            margin-top: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Check and load models
if not (os.path.exists('./Dataset/final_data.csv') and os.path.exists('./Models/cosine_sim.npy')):
    warn = st.warning('Models not found! \n Running the notebooks to create models.')
    pm.execute_notebook(
        './BookRec.ipynb'
    )
    warn.empty()
else:
    st.success('Models already exist!', icon="✅")

ncf_model = NCF(num_users, num_items)
cnn_model = TextCNN(max_words, max_len=max_len)

# Function to load the models
@st.cache_resource()
def load_models():
    cosine_sim = np.load('./Models/cosine_sim.npy')
    ncf_model.load_state_dict(torch.load('./Models/ncf_model.pth'))
    cnn_model.load_state_dict(torch.load('./Models/text_cnn_model.pth'))
    df = pd.read_csv("./Dataset/final_data_with_ratings.csv")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    st.success('Models loaded successfully!', icon="✅")
    return cosine_sim, df, ncf_model, cnn_model, tokenizer

cosine_sim, final_data, ncf_model, cnn_model, tokenizer = load_models()

# Get the list of book titles
options = final_data['Title'].values.tolist()

# Create the Streamlit app
def main():
    # Set the app title
    st.markdown('<div class="title">📚 BookRec: An AI Book Recommender</div>', unsafe_allow_html=True)

    # Add a centered dropdown to the main content
    selected_option = st.selectbox('Select a book to get recommendations', pd.Series(options).sort_values().unique())

    st.markdown(f'<div class="subtitle">You selected: {selected_option}</div>', unsafe_allow_html=True)

    # Recommendation function
    def hybrid_recommendation(book_title, df, cosine_sim, ncf_model, cnn_model, tokenizer, max_len=200):
        def recommend_books_cosine(book_title, final_data, cosine_sim):
            if not final_data.empty:
                idx = final_data[final_data['Title'] == book_title].index
                if len(idx) > 0:
                    idx = idx[0]
                    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
                    book_indices = [i[0] for i in sim_scores]
                    return final_data[['Title', 'Image', 'Author', 'Pages']].iloc[book_indices]
                else:
                    return "Book not found"
            else:
                return "No data available"

        cosine_recs = recommend_books_cosine(book_title, df, cosine_sim)
        user_id = df['user_id'].iloc[0]
        book_ids = [df[df['Title'] == title]['ISBN'].values[0] for title in cosine_recs['Title']]

        ncf_recs = []
        for book_id in book_ids:
            user_tensor = torch.tensor([int(user_id)], dtype=torch.long)
            book_tensor = torch.tensor([int(book_id)], dtype=torch.long)
            with torch.no_grad():
                rating_pred = ncf_model(user_tensor, book_tensor).item()
            ncf_recs.append((book_id, rating_pred))
        ncf_recs = sorted(ncf_recs, key=lambda x: x[1], reverse=True)

        cnn_recs = []
        for book_id in book_ids:
            book_desc = df[df['ISBN'] == book_id]['Desc'].values[0]
            inputs = tokenizer(book_desc, return_tensors='pt', truncation=True, padding='max_length', max_length=max_len)
            input_tensor = inputs['input_ids'].clamp(max=max_words-1)  # Ensure indices are within range
            with torch.no_grad():
                rating_pred = cnn_model(input_tensor).item()
            cnn_recs.append((book_id, rating_pred))
        cnn_recs = sorted(cnn_recs, key=lambda x: x[1], reverse=True)

        combined_recs = list(set([book for book, _ in cnn_recs]))
        final_recs = [(df[df['ISBN'] == book_id]['Title'].values[0],
                       df[df['ISBN'] == book_id]['Image'].values[0],
                       df[df['ISBN'] == book_id]['Author'].values[0],
                       df[df['ISBN'] == book_id]['Pages'].values[0])
                      for book_id in combined_recs]
        return pd.DataFrame(final_recs, columns=['Title', 'Image', 'Author', 'Pages'])

    book = hybrid_recommendation(selected_option, final_data, cosine_sim, ncf_model, cnn_model, tokenizer)

    # Display book recommendations in rows of 3
    st.subheader('Recommended Books')

    for i in range(0, len(book), 3):
        cols = st.columns(3, gap='large')
        for j in range(3):
            if i + j < len(book):
                with cols[j]:
                    st.markdown(f"""
                        <div class="book-card">
                            <img src="{book.iloc[i + j, 1]}" alt="{book.iloc[i + j, 0]}" class="book-image">
                            <p class="title">{book.iloc[i + j, 0]}</p>
                            <p>Author: {book.iloc[i + j, 2]}</p>
                            <p>Pages: {book.iloc[i + j, 3]}</p>
                        </div>
                    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
