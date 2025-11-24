from flask import Flask, request, render_template
import pandas as pd
import pickle

# Load dataset once when the server starts
data = pd.read_csv("IMDB-Movie-Dataset(2023-1951) (2).csv")

data = data.dropna()

# Initialize the Flask app
app = Flask(__name__)
# app.config['BACKGROUND_IMAGE'] = 'static/background.jpg'

# Load the Filter object from the pickle file
with open('filter_obj.pkl', 'rb') as file:
    filter_obj = pickle.load(file)

# Load the KNN object from the pickle file
with open('knn_obj.pkl', 'rb') as file:
    knn_obj = pickle.load(file)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    print("Inside recommend")
    cast = str(request.form.get('cast', 'none'))
    director = str(request.form.get('director', 'none'))
    genre = str(request.form.get('genre', 'none'))
    year = "none"
    
    top_movies_str = filter_obj.TOP(cast=cast, director=director, genre=genre, year=year, data=data)
    
    split_by_question_mark = top_movies_str.split('?')
    result_2d_list = []

    for string_with_pipe in split_by_question_mark:
        split_by_pipe = string_with_pipe.split(' | ')
        result_2d_list.append(split_by_pipe)
    
    return render_template('index1.html', movies_info=result_2d_list)

@app.route('/tell', methods=['POST'])
def tell():
    movie = str(request.form.get('movie', 'none'))

    suggested_movie = knn_obj.predict(data=data, movie_name=movie)
    
    split_by_question_mark = suggested_movie.split('?')
    result_2d_list = []

    for string_with_pipe in split_by_question_mark:
        split_by_pipe = string_with_pipe.split(' | ')
        result_2d_list.append(split_by_pipe)

    return render_template('index1.html', result=result_2d_list)

if __name__ == '__main__':
    app.run(debug=True)
