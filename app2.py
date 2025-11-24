import gradio as gr
import pandas as pd
import pickle

movies = pickle.load(open('movies.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
vectorizer = pickle.load(open('model.pkl', 'rb'))

# Recommendation function
def recommend_movie(movie_name):
    if movie_name not in movies['movie_name'].values:
        return "Movie not found."
    index = movies[movies['movie_name'] == movie_name].index[0]
    from sklearn.metrics.pairwise import cosine_similarity
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    top_10 = [movies.iloc[i[0]]['movie_name'] for i in distances[1:11]]
    return "\n".join(top_10)

# Gradio interface
app = gr.Interface(
    fn=recommend_movie,
    inputs=gr.Textbox(label="Enter a movie name:"),
    outputs=gr.Textbox(label="Recommended Movies", lines=8, max_lines=12),
    title="Movie Recommendation System",
    description="Content-based filtering using cosine similarity and TMDb metadata."
)

app.launch(debug=True)
