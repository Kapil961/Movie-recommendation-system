def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    user_movie_matrix = user_movie_matrix.fillna(0)

    return user_movie_matrix


def create_movie_user_matrix(user_movie_matrix):
    movie_user_matrix = user_movie_matrix.T

    return movie_user_matrix