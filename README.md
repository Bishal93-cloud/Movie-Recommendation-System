# Movie Recommendation System

A simple movie recommender in Python using collaborative filtering on the MovieLens 100k dataset.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Bishal93-cloud/Movie-Recommendation-System.git](https://github.com/Bishal93-cloud/Movie-Recommendation-System.git)
    cd Movie-Recommendation-System
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the Dataset:**
    * This project uses the MovieLens 100k dataset.
    * Download it from here: [https://files.grouplens.org/datasets/movelens/ml-100k.zip](https://files.grouplens.org/datasets/movelens/ml-100k.zip)
    * Unzip the file and place `u.data` and `u.item` in the main project folder (the same folder as `recommender.py`).

4.  **Run the script:**
    ```bash
    python recommender.py
    ```

## About the Code

This script uses the `scikit-learn` library to find the K-Nearest Neighbors for a given movie. Similarity is measured using the cosine distance between movie rating vectors.

This script uses the `scikit-learn` library to find the K-Nearest Neighbors for a given movie. Similarity is measured using the cosine distance between movie rating vectors.
