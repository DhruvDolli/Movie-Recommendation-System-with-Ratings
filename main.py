import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv('movies2.csv')
ratings = pd.read_csv('ratings.csv')  # Assuming you have a ratings.csv file with movie ratings

# Merge movies with ratings
movies_with_ratings = pd.merge(movies, ratings, on='movieId', how='left')

# Combine title and genre to create tags
movies['tags'] = movies['genres'] + ' ' + movies['title']

# Select relevant columns
new_df = movies[['title']]

# Initialize CountVectorizer
cv = CountVectorizer(max_features=9743, stop_words='english')

# Fit and transform the titles
vec = cv.fit_transform(new_df['title'].values.astype('U')).toarray()

# Compute cosine similarity
sim = cosine_similarity(vec)

# Define a function to recommend movies
def recommend(movie_title):
    index = new_df[new_df['title'] == movie_title].index[0]
    distance = sorted(list(enumerate(sim[index])), reverse=True, key=lambda x: x[1])
    recommendations = []
    for i in distance[1:6]:
        movie = new_df.iloc[i[0]]
        rating = movies_with_ratings[movies_with_ratings['title'] == movie['title']]['rating'].mean()
        recommendations.append((movie['title'], rating))
    return recommendations

# Streamlit frontend

#sidebar
st.sidebar.title("Dashboard")
app = st.sidebar.selectbox("Select Page", ["FlickAI Overview", "FlickAI"])

#First Page
if(app == "FlickAI Overview"):
    st.header("FlickAI:")
    st.write("FlickAI is an advanced movie recommendation system designed to provide personalized movie suggestions based on user preferences and historical viewing behavior. Leveraging state-of-the-art machine learning algorithms, FlickAI delivers tailored movie recommendations that enhance user satisfaction and engagement with movie-watching experiences.")
    img = "AI & ML home page.jpg"
    st.image(img)

#Second Page
elif(app == "FlickAI"):
    st.header("FlickAI  🤖")
    st.title('Movie Recommendation System')
    movie_input = st.text_input('Enter a movie title:', 'Exorcist III, The (1990)')
    if st.button('Recommend'):
        recommendations = recommend(movie_input)
        st.write('### Recommendations:')
        for title, rating in recommendations:
            st.write(f'- **{title}**    (Rating: {int(rating)}/5)')