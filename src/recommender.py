from sklearn.neighbors import NearestNeighbors


def train_recommender(movie_user_matrix):
    """
    Train KNN recommendation model.
    """

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(movie_user_matrix)

    return model


def get_movie_recommendations(
    movie_name,
    movies,
    movie_user_matrix,
    model,
    n_recommendations=5
):
    """
    Recommend similar movies.
    """

    matching_movies = movies[
        movies["title"].str.contains(
            movie_name,
            case=False,
            na=False
        )
    ]

    if matching_movies.empty:
        return "Movie not found."

    movie_id = matching_movies.iloc[0]["movieId"]

    movie_title = matching_movies.iloc[0]["title"]

    if movie_id not in movie_user_matrix.index:
        return "Movie has no ratings data."

    movie_index = movie_user_matrix.index.get_loc(
        movie_id
    )

    distances, indices = model.kneighbors(
        movie_user_matrix.iloc[movie_index]
        .values.reshape(1, -1),
        n_neighbors=n_recommendations + 1
    )

    recommendations = []

    for i in range(
        1,
        len(indices.flatten())
    ):
        recommended_movie_id = (
            movie_user_matrix.index[
                indices.flatten()[i]
            ]
        )

        recommended_title = movies[
            movies["movieId"]
            == recommended_movie_id
        ]["title"].values[0]

        recommendations.append(
            recommended_title
        )

    return movie_title, recommendations