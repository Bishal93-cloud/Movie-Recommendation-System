import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

def create_movie_recommendation_system():
    """
    Loads MovieLens data, builds a k-NN model, and sets up for recommendations.
    """
    # --- Step 1: Load and Prepare the Data ---

    # Define column names for the ratings data (u.data)
    # It's a tab-separated file with user_id, item_id, rating, timestamp
    script_dir = os.path.dirname(__file__) 

# Create the full path to the data files
    u_data_path = os.path.join(script_dir, 'u.data')
    u_item_path = os.path.join(script_dir, 'u.item')

# Read the files using their full paths
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp'] # This line is for u.data
    ratings = pd.read_csv(u_data_path, sep='\t', names=r_cols, encoding='latin-1')

# This is the line you asked about, it MUST be here for u.item
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv(u_item_path, sep='|', names=m_cols, encoding='latin-1')


    # Merge the ratings and movies dataframes into one
    df = pd.merge(ratings, movies, on='movie_id')

    # --- Step 2: Create the User-Item Matrix ---
    
    # A user-item matrix is a table where rows are users and columns are movies.
    # The values are the ratings the user gave to that movie.
    # 'NaN' means the user hasn't rated the movie.
    movie_matrix = df.pivot_table(index='title', columns='user_id', values='rating').fillna(0)

    # Convert the pandas dataframe into a sparse matrix for efficiency
    # Sparse matrices are better when most elements are zero.
    movie_sparse_matrix = csr_matrix(movie_matrix.values)

    # --- Step 3: Build the k-NN Model ---

    # We use 'cosine' similarity to find movies that are "close" to each other
    # in terms of user ratings.
    # 'brute' algorithm will check all possible pairs.
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(movie_sparse_matrix)
    
    return model_knn, movie_matrix


def get_movie_recommendations(movie_name, model, matrix, n_recommendations=5):
    """
    Finds movies similar to a given movie.
    """
    if movie_name not in matrix.index:
        return f"Movie '{movie_name}' not found in the dataset. Please check the spelling."

    # Get the index of the movie in our matrix
    movie_index = matrix.index.get_loc(movie_name)
    
    # Find the nearest neighbors (the most similar movies)
    # It returns two lists: distances and indices of the neighbors
    distances, indices = model.kneighbors(matrix.iloc[movie_index, :].values.reshape(1, -1), n_neighbors=n_recommendations + 1)
    
    # We get n+1 movies because the first result is always the movie itself.
    # We will skip the first one (distance = 0) to get the actual recommendations.
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommended_movie_name = matrix.index[indices.flatten()[i]]
        recommendations.append(recommended_movie_name)
        
    return recommendations


# --- Main Execution ---

if __name__ == '__main__':
    # Build the model and get the matrix
    knn_model, movie_data_matrix = create_movie_recommendation_system()

    # --- Get Recommendations for a Movie ---
    # You can change the movie title here to get different recommendations
    movie_to_recommend_for = 'Star Wars (1977)'
    
    print(f"✨ Recommendations based on your interest in '{movie_to_recommend_for}':")
    
    recommendations = get_movie_recommendations(
        movie_name=movie_to_recommend_for,
        model=knn_model,
        matrix=movie_data_matrix,
        n_recommendations=5
    )

    if isinstance(recommendations, list):
        for i, movie in enumerate(recommendations):
            print(f"{i+1}: {movie}")
    else:
        print(recommendations)