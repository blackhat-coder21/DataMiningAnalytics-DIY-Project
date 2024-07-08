import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Load data with specified encoding
movies_df = pd.read_csv('Movies.csv', encoding='latin1')
ratings_df = pd.read_csv('Ratings.csv', encoding='latin1')
users_df = pd.read_csv('Users.csv', encoding='latin1')

# Extract year of release from the movie title
movies_df['YearOfRelease'] = movies_df['Title'].str.extract(r'\((\d{4})\)')
movies_df['YearOfRelease'] = pd.to_numeric(movies_df['YearOfRelease'], errors='coerce')

# Remove rows with NaN YearOfRelease
movies_df = movies_df.dropna(subset=['YearOfRelease'])
movies_df['YearOfRelease'] = movies_df['YearOfRelease'].astype(int)

# Clean and preprocess data
movies_df['Categories'] = movies_df['Category'].str.split('|')
ratings_with_movies = pd.merge(ratings_df, movies_df, on='MovieID')
ratings_with_users = pd.merge(ratings_with_movies, users_df, on='UserID')

# Query i: Total number of movies released in each year
movies_per_year = movies_df.groupby('YearOfRelease').size().reset_index(name='Count')

# Query ii: Find the movie category having highest ratings in each year
average_ratings = ratings_with_movies.groupby(['MovieID', 'YearOfRelease', 'Category'])['Rating'].mean().reset_index()
highest_rated_categories = average_ratings.loc[average_ratings.groupby('YearOfRelease')['Rating'].idxmax()]

# Query iii: Movie category and age group wise likings
age_group_likings = ratings_with_users.groupby(['Age', 'Category']).size().reset_index(name='Count')
age_group_likings = age_group_likings.loc[age_group_likings.groupby('Age')['Count'].idxmax()]

# Query iv: Clustering models for movie category and age group wise likings
category_age_group_df = ratings_with_users[['Category', 'Age']]
category_age_group_df['Category'] = category_age_group_df['Category'].astype('category').cat.codes
scaler = StandardScaler()
category_age_group_scaled = scaler.fit_transform(category_age_group_df)
kmeans_age_group = KMeans(n_clusters=5, random_state=42)
kmeans_age_group.fit(category_age_group_scaled)
category_age_group_df['Cluster'] = kmeans_age_group.labels_

# Query v: Year wise count of movies released
yearly_movie_count = movies_df.groupby('YearOfRelease').size().reset_index(name='Count')

# Query vi: Year wise, category wise count of movies released
year_category_movie_count = movies_df.explode('Categories').groupby(['YearOfRelease', 'Categories']).size().reset_index(name='Count')

# Query vii: Clustering methods to segregate movie category and occupation of users
category_occupation_df = ratings_with_users[['Category', 'Occupation']]
category_occupation_df['Category'] = category_occupation_df['Category'].astype('category').cat.codes
category_occupation_scaled = scaler.fit_transform(category_occupation_df)
kmeans_occupation = KMeans(n_clusters=5, random_state=42)
kmeans_occupation.fit(category_occupation_scaled)
category_occupation_df['Cluster'] = kmeans_occupation.labels_

# Query viii: Refine the model by including age group
category_occupation_age_df = ratings_with_users[['Category', 'Occupation', 'Age']]
category_occupation_age_df['Category'] = category_occupation_age_df['Category'].astype('category').cat.codes
category_occupation_age_scaled = scaler.fit_transform(category_occupation_age_df)
kmeans_occupation_age = KMeans(n_clusters=5, random_state=42)
kmeans_occupation_age.fit(category_occupation_age_scaled)
category_occupation_age_df['Cluster'] = kmeans_occupation_age.labels_

# Query ix: Predict movie likings based on category, age group, and occupation
def predict_movie_liking(category, occupation, age):
    category_code = pd.Series([category]).astype('category').cat.codes[0]
    occupation_age_data = scaler.transform([[category_code, occupation, age]])
    cluster = kmeans_occupation_age.predict(occupation_age_data)[0]
    return category_occupation_age_df[category_occupation_age_df['Cluster'] == cluster]

# Streamlit UI
st.title("Movie Analysis and Prediction System")

query_option = st.sidebar.selectbox("Select Query", [
    "Total number of movies released in each year",
    "Movie category having highest ratings in each year",
    "Movie category and age group wise likings",
    "Year wise count of movies released",
    "Year wise, category wise count of movies released",
    "Clustering movie category and occupation of users",
    "Refine model with age group",
    "Predict movie likings"
])

if query_option == "Total number of movies released in each year":
    st.write(movies_per_year)
elif query_option == "Movie category having highest ratings in each year":
    st.write(highest_rated_categories)
elif query_option == "Movie category and age group wise likings":
    st.write(age_group_likings)
elif query_option == "Year wise count of movies released":
    st.write(yearly_movie_count)
elif query_option == "Year wise, category wise count of movies released":
    st.write(year_category_movie_count)
elif query_option == "Clustering movie category and occupation of users":
    st.write(category_occupation_df)
elif query_option == "Refine model with age group":
    st.write(category_occupation_age_df)
elif query_option == "Predict movie likings":
    category = st.text_input("Enter movie category")
    occupation = st.number_input("Enter occupation code", min_value=0, max_value=20, step=1)
    age = st.number_input("Enter age group code", min_value=1, max_value=56, step=1)
    if st.button("Predict"):
        result = predict_movie_liking(category, occupation, age)
        st.write(result)
