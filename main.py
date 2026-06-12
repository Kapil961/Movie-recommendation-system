import pandas as pd

from src.data_loader import load_data

from src.preprocessing import (
    create_user_movie_matrix,
    create_movie_user_matrix
)

from src.recommender import (
    train_recommender,
    get_movie_recommendations
)

from src.evaluation import (
    calculate_rmse,
    calculate_coverage
)


# ==========================
# Load Dataset
# ==========================

movies, ratings = load_data()

print("Movies Shape:", movies.shape)

print("Ratings Shape:", ratings.shape)


# ==========================
# Create Matrices
# ==========================

user_movie_matrix = (
    create_user_movie_matrix(
        ratings
    )
)

movie_user_matrix = (
    create_movie_user_matrix(
        user_movie_matrix
    )
)

print("\nUser-Movie Matrix Shape:")

print(user_movie_matrix.shape)

print("\nMovie-User Matrix Shape:")

print(movie_user_matrix.shape)


# ==========================
# Train Recommendation Model
# ==========================

model = train_recommender(
    movie_user_matrix
)


# ==========================
# Get Recommendations
# ==========================

movie_name = "Toy Story"

result = get_movie_recommendations(
    movie_name,
    movies,
    movie_user_matrix,
    model,
    n_recommendations=5
)

print("\n=== Recommendations ===")

if isinstance(result, str):

    print(result)

else:

    selected_movie, recommendations = result

    print(
        "Selected Movie:",
        selected_movie
    )

    for i, movie in enumerate(
        recommendations,
        start=1
    ):
        print(f"{i}. {movie}")


# ==========================
# Save Recommendations
# ==========================

if not isinstance(result, str):

    recommendation_df = pd.DataFrame({
        "Selected Movie":
            [selected_movie]
            * len(recommendations),

        "Recommended Movies":
            recommendations
    })

    recommendation_df.to_csv(
        "outputs/reports/recommendations.csv",
        index=False
    )

    print(
        "\nRecommendations saved successfully."
    )


# ==========================
# Evaluation
# ==========================

rmse = calculate_rmse(
    ratings
)

coverage = calculate_coverage(
    movie_user_matrix
)

print("\n=== Evaluation Metrics ===")

print(f"RMSE: {rmse:.4f}")

print(
    f"Coverage: {coverage:.4f}"
)


# ==========================
# Save Evaluation Report
# ==========================

evaluation_report = pd.DataFrame({
    "Metric": [
        "RMSE",
        "Coverage"
    ],

    "Value": [
        rmse,
        coverage
    ]
})

evaluation_report.to_csv(
    "outputs/reports/evaluation_report.csv",
    index=False
)

print(
    "Evaluation report saved successfully."
)