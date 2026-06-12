import numpy as np
from sklearn.metrics import mean_squared_error


def calculate_rmse(ratings):
    y_true = ratings["rating"]

    y_pred = [ratings["rating"].mean()] * len(ratings)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    return rmse


def calculate_coverage(movie_user_matrix):
    total_movies = movie_user_matrix.shape[0]

    movies_with_ratings = (
        movie_user_matrix.sum(axis=1) > 0
    ).sum()

    coverage = movies_with_ratings / total_movies

    return coverage