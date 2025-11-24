import gradio as gr
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
vectorizer = pickle.load(open('model.pkl', 'rb'))

# Recommendation function
def recommend_movie(movie_name):
    movie_name = movie_name.strip().lower()

    if movie_name not in movies['movie_name'].str.lower().values:
        return "Movie not found"
    
    index = movies[movies['movie_name'].str.lower() == movie_name].index[0]

    # Compute cosine similarity using the stored vectorizer
    # Re-transform the same tag using vectorizer for consistency
    movie_vector = vectorizer.transform([movies.iloc[index]['tags']])
    cosine_similarities = cosine_similarity(movie_vector, vectorizer.transform(movies['tags'])).flatten()

    # Sort by similarity score
    top_indices = cosine_similarities.argsort()[-11:][::-1]  # top 10 similar (excluding itself)
    top_movies = [movies.iloc[i]['movie_name'] for i in top_indices if i != index][:10]

    return "\n".join(top_movies)

# Gradio interface
app = gr.Interface(
    fn=recommend_movie,
    inputs=gr.Textbox(label="Enter a movie name:"),
    outputs=gr.Textbox(label="Recommended Movies", lines=8, max_lines=12),
    title="Movie Recommendation System",
    description="Content-based filtering using cosine similarity and TMDb metadata."
)

app.launch(debug=True)
