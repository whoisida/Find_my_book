import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model
model = SentenceTransformer('cointegrated/rubert-tiny2')

# Load dataset
databook = pd.read_csv('/Users/idaklimanova/ds_bootcamp/ds-phase-2/project4/dataset_all.csv')

# Load saved embeddings
embeddings = torch.load('book_embeddings3.pth')

# Function to get most similar books
def get_most_similar_books(user_query, embeddings):
    # Encode user query
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    user_query = util.normalize_embeddings(query_embedding.unsqueeze(0))  # Add unsqueeze to add batch dimension
    
    # Convert torch tensors to numpy arrays
    user_query_np = user_query.cpu().detach().numpy()
    embeddings_np = embeddings.cpu().detach().numpy()
    
    # Calculate similarity using cosine similarity
    similarities = cosine_similarity(user_query_np, embeddings_np)
    
    # Get indices of most similar books
    indices = np.argsort(similarities[0])[::-1][:5]  # Get top 5 indices with highest cosine similarity
    return indices


def main():
    st.title("Рекомендации книг")

    # Get user query
    user_query = st.text_input("Введите описание книги:")

    button = st.button('Отправить запрос', type="primary")

    # Show recommended books
    if user_query:
        st.write("Наиболее подходящие книги для вас:")
        similar_books_indices = get_most_similar_books(user_query, embeddings)
        for idx in similar_books_indices:
            book = databook.iloc[idx]

            # Display book cover image and description side by side
            col1, col2 = st.columns([1, 4])  # Layout ratio 1:4
            with col1:
                if pd.notna(book['image_url']):
                    try:
                        image = Image.open(requests.get(book['image_url'], stream=True).raw)
                        st.image(image, caption='Обложка книги', width=130)
                    except Exception as e:
                        st.write("Ошибка при загрузке изображения:", e)
            with col2:
                st.subheader(book['title'])
                st.write(f"Автор: {book['author']}")
                st.write(f"Описание: {book['annotation']}")

if __name__ == "__main__":
    main()
