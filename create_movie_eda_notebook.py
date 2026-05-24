from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from nbformat import v4 as nbf


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "MovieDB_EDA_Intern_Project.ipynb"


def markdown_cell(text: str):
    return nbf.new_markdown_cell(dedent(text).strip())


def code_cell(code: str):
    return nbf.new_code_cell(dedent(code).strip())


cells = [
    markdown_cell(
        """
        # MovieDB Exploratory Data Analysis

        **Dataset:** MovieDB movie dataset

        This project explores movie trends using Python, Pandas, NumPy, Matplotlib, and Seaborn.
        """
    ),
    markdown_cell(
        """
        ## 1. Project Introduction

        I used Exploratory Data Analysis to understand the movie dataset before drawing any conclusions. The focus is on movie popularity, ratings, genres, languages, and release trends.
        """
    ),
    markdown_cell(
        """
        ## 2. Problem Statement

        The goal is to answer a few simple questions from the dataset:

        - Which genres appear most often?
        - Which original languages are most common?
        - How have movie releases changed over the years?
        - Is there a relationship between popularity, vote count, and vote average?
        - Which types of movies seem to perform better?
        """
    ),
    markdown_cell("## 3. Import Libraries"),
    code_cell(
        """
        # Data handling
        import pandas as pd
        import numpy as np

        # Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Helper libraries
        from pathlib import Path
        import io
        from IPython.display import display

        %matplotlib inline

        # Cleaner table and chart display
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        sns.set_theme(style="whitegrid", palette="Set2")
        plt.rcParams["figure.figsize"] = (10, 5)
        """
    ),
    markdown_cell("## 4. Load Dataset"),
    code_cell(
        """
        # Load the dataset. The Python engine handles long overview text more safely.
        possible_paths = [
            Path("mymoviedb.csv"),
            Path("mymoviedb(1).csv"),
            Path("../mymoviedb.csv"),
            Path("../mymoviedb(1).csv"),
            Path("/mnt/data/mymoviedb.csv"),
            Path("/mnt/data/mymoviedb(1).csv"),
        ]

        dataset_path = None
        for path in possible_paths:
            if path.exists():
                dataset_path = path
                break

        if dataset_path is None:
            raise FileNotFoundError("Dataset file was not found. Please place mymoviedb.csv near this notebook.")

        movies_raw = pd.read_csv(dataset_path, engine="python")

        print("Dataset loaded successfully.")
        """
    ),
    markdown_cell("## 5. Dataset Overview"),
    code_cell(
        """
        # Display the first five rows
        print("First 5 rows:")
        display(movies_raw.head())

        # Display the last five rows
        print("Last 5 rows:")
        display(movies_raw.tail())

        # Display shape of dataset
        print("Dataset shape:", movies_raw.shape)

        # Display column names
        print("Column names:")
        print(movies_raw.columns.tolist())

        # Display dataset information clearly
        print("\\nDataset info:")
        buffer = io.StringIO()
        movies_raw.info(buf=buffer)
        print(buffer.getvalue())

        # Display descriptive statistics
        print("Descriptive statistics:")
        display(movies_raw.describe(include="all").T)
        """
    ),
    markdown_cell("## 6. Data Cleaning"),
    markdown_cell(
        """
        I cleaned the dataset with the steps needed for reliable analysis:

        - Standardize column names
        - Check missing values
        - Remove duplicate rows
        - Convert date and numeric columns
        - Handle missing or invalid rows
        - Detect outliers using boxplots
        - Handle simple outliers using the IQR method
        """
    ),
    code_cell(
        """
        # Create a copy so the original dataset remains unchanged
        movies = movies_raw.copy()

        # Standardize column names: lowercase and replace spaces with underscores
        old_columns = movies.columns.tolist()
        movies.columns = (
            movies.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        column_changes = pd.DataFrame({
            "old_column_name": old_columns,
            "new_column_name": movies.columns.tolist()
        })
        display(column_changes)

        # Check missing values before cleaning
        print("Missing values before cleaning:")
        display(movies.isnull().sum().to_frame("missing_values"))

        # Check duplicate rows
        duplicate_count = movies.duplicated().sum()
        print("Duplicate rows before cleaning:", duplicate_count)

        # Remove duplicate rows
        movies = movies.drop_duplicates()
        print("Shape after removing duplicates:", movies.shape)
        """
    ),
    code_cell(
        """
        # Convert release_date to datetime format
        movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")

        # Convert numeric columns from object/string to numeric values
        numeric_columns = ["popularity", "vote_count", "vote_average"]
        for column in numeric_columns:
            movies[column] = pd.to_numeric(movies[column], errors="coerce")

        print("Data types after conversion:")
        display(movies.dtypes.to_frame("data_type"))

        print("Missing values after type conversion:")
        display(movies.isnull().sum().to_frame("missing_values"))
        """
    ),
    code_cell(
        """
        # Remove rows missing fields needed for analysis.
        important_columns = [
            "release_date",
            "title",
            "popularity",
            "vote_count",
            "vote_average",
            "original_language",
            "genre",
            "poster_url",
        ]

        rows_before = movies.shape[0]
        movies = movies.dropna(subset=important_columns).copy()

        # Fill any missing overview text.
        movies["overview"] = movies["overview"].fillna("No overview available")

        rows_after = movies.shape[0]
        print("Rows removed due to missing or invalid important values:", rows_before - rows_after)
        print("Final shape after missing value handling:", movies.shape)

        print("Missing values after cleaning:")
        display(movies.isnull().sum().to_frame("missing_values"))
        """
    ),
    code_cell(
        """
        # Detect outliers using boxplots
        outlier_columns = ["popularity", "vote_count", "vote_average"]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for index, column in enumerate(outlier_columns):
            sns.boxplot(y=movies[column], ax=axes[index], color="#8ecae6")
            axes[index].set_title(f"Boxplot of {column.replace('_', ' ').title()}")
            axes[index].set_ylabel(column.replace("_", " ").title())

        plt.suptitle("Outlier Detection Before IQR Treatment", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

        # Create an outlier report using the IQR method
        outlier_report = []
        outlier_limits = {}

        for column in outlier_columns:
            q1 = movies[column].quantile(0.25)
            q3 = movies[column].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr

            outlier_count = ((movies[column] < lower_limit) | (movies[column] > upper_limit)).sum()

            outlier_limits[column] = (lower_limit, upper_limit)
            outlier_report.append({
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_limit": lower_limit,
                "upper_limit": upper_limit,
                "outliers_found": int(outlier_count),
            })

        outlier_report_df = pd.DataFrame(outlier_report)
        display(outlier_report_df)
        """
    ),
    code_cell(
        """
        # Cap extreme values with the IQR method.
        eda_df = movies.copy()

        for column in outlier_columns:
            lower_limit, upper_limit = outlier_limits[column]
            eda_df[column] = eda_df[column].clip(lower=lower_limit, upper=upper_limit)

        # Check boxplots again after capping
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for index, column in enumerate(outlier_columns):
            sns.boxplot(y=eda_df[column], ax=axes[index], color="#90be6d")
            axes[index].set_title(f"Boxplot of {column.replace('_', ' ').title()} After Capping")
            axes[index].set_ylabel(column.replace("_", " ").title())

        plt.suptitle("Outlier Boxplots After IQR Treatment", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

        print("Outliers were handled using IQR capping.")
        print("Original cleaned data shape:", movies.shape)
        print("EDA data shape after capping:", eda_df.shape)
        """
    ),
    markdown_cell(
        """
        **Observation:** Popularity and vote count have several high values, which is expected because a few movies receive much more attention than the rest. I used IQR capping so these extreme values do not dominate the analysis.
        """
    ),
    markdown_cell("## 7. Feature Engineering"),
    code_cell(
        """
        # Create useful features from existing columns.
        eda_df["release_year"] = eda_df["release_date"].dt.year
        eda_df["release_month"] = eda_df["release_date"].dt.month

        # Text length features
        eda_df["title_length"] = eda_df["title"].astype(str).str.len()
        eda_df["overview_length"] = eda_df["overview"].astype(str).str.len()

        # Use the first listed genre as the primary genre.
        eda_df["primary_genre"] = eda_df["genre"].str.split(",").str[0].str.strip()

        print("New features created:")
        print(["release_year", "release_month", "title_length", "overview_length", "primary_genre"])

        display(eda_df[[
            "title",
            "release_date",
            "release_year",
            "release_month",
            "title_length",
            "overview_length",
            "primary_genre",
        ]].head())
        """
    ),
    markdown_cell("## 8. Exploratory Data Analysis"),
    code_cell(
        """
        # Summary statistics for numerical columns
        numerical_features = [
            "popularity",
            "vote_count",
            "vote_average",
            "release_year",
            "release_month",
            "title_length",
            "overview_length",
        ]

        print("Summary statistics for numerical features:")
        display(eda_df[numerical_features].describe().T)

        # Summary for categorical columns
        print("Summary for categorical features:")
        display(eda_df[["original_language", "primary_genre"]].describe().T)
        """
    ),
    code_cell(
        """
        # Distribution analysis
        print("Distribution summary for key movie metrics:")
        display(eda_df[["popularity", "vote_average", "vote_count"]].agg(["mean", "median", "std", "min", "max"]).T)

        # Category analysis: top genres and languages
        all_genres = eda_df["genre"].str.split(",").explode().str.strip()
        top_genres = all_genres.value_counts().head(10)
        top_languages = eda_df["original_language"].value_counts().head(10)

        print("Top 10 genres:")
        display(top_genres.to_frame("movie_count"))

        print("Top 10 original languages:")
        display(top_languages.to_frame("movie_count"))

        # Trend analysis: movies released by year
        yearly_movies = (
            eda_df.groupby("release_year")
            .size()
            .reset_index(name="movie_count")
            .sort_values("release_year")
        )

        print("Recent yearly release trend:")
        display(yearly_movies.tail(10))
        """
    ),
    markdown_cell("## 9. Visualizations"),
    code_cell(
        """
        # Histogram: popularity distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(eda_df["popularity"], bins=30, kde=True, color="#219ebc")
        plt.title("Popularity Distribution")
        plt.xlabel("Popularity")
        plt.ylabel("Number of Movies")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Most movies sit in the lower popularity range, while a small number of titles stand out with much higher popularity.
        """
    ),
    code_cell(
        """
        # Histogram: vote average distribution
        plt.figure(figsize=(10, 5))
        sns.histplot(eda_df["vote_average"], bins=25, kde=True, color="#ffb703")
        plt.title("Vote Average Distribution")
        plt.xlabel("Vote Average")
        plt.ylabel("Number of Movies")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Most vote averages are around the middle range, so the dataset is mostly made up of moderately rated movies.
        """
    ),
    code_cell(
        """
        # Countplot: top genres
        top_genre_names = top_genres.index.tolist()
        genre_plot_df = pd.DataFrame({"genre": all_genres})
        genre_plot_df = genre_plot_df[genre_plot_df["genre"].isin(top_genre_names)]

        plt.figure(figsize=(11, 6))
        sns.countplot(data=genre_plot_df, y="genre", hue="genre", order=top_genre_names, palette="viridis", legend=False)
        plt.title("Top 10 Movie Genres")
        plt.xlabel("Number of Movies")
        plt.ylabel("Genre")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Drama, Comedy, Action, and Thriller appear most often, which makes them the main genres in this dataset.
        """
    ),
    code_cell(
        """
        # Bar chart: average popularity by primary genre
        avg_popularity_by_genre = (
            eda_df.groupby("primary_genre")["popularity"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=avg_popularity_by_genre,
            x="popularity",
            y="primary_genre",
            hue="primary_genre",
            palette="mako",
            legend=False,
        )
        plt.title("Average Popularity by Primary Genre")
        plt.xlabel("Average Popularity")
        plt.ylabel("Primary Genre")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Action, Animation, and Science Fiction movies show stronger average popularity than many other genres.
        """
    ),
    code_cell(
        """
        # Countplot: most common original languages
        language_order = top_languages.index.tolist()

        plt.figure(figsize=(10, 5))
        sns.countplot(
            data=eda_df[eda_df["original_language"].isin(language_order)],
            x="original_language",
            hue="original_language",
            order=language_order,
            palette="crest",
            legend=False,
        )
        plt.title("Top 10 Original Languages")
        plt.xlabel("Original Language")
        plt.ylabel("Number of Movies")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** English is the dominant original language. Japanese, Spanish, and French appear next but with much smaller counts.
        """
    ),
    code_cell(
        """
        # Line chart: movies released per year
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=yearly_movies, x="release_year", y="movie_count", marker="o", color="#fb8500")
        plt.title("Movies Released per Year")
        plt.xlabel("Release Year")
        plt.ylabel("Number of Movies")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Movie releases rise clearly from the 2000s onward. The last few years may be lower because the dataset may not be fully updated.
        """
    ),
    code_cell(
        """
        # Scatter plot: popularity vs vote average
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=eda_df,
            x="vote_average",
            y="popularity",
            hue="primary_genre",
            alpha=0.65,
            legend=False,
        )
        plt.title("Popularity vs Vote Average")
        plt.xlabel("Vote Average")
        plt.ylabel("Popularity")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Popularity and rating do not move together strongly. A movie can be popular without having the highest rating.
        """
    ),
    code_cell(
        """
        # Boxplot: vote average by top primary genres
        top_primary_genres = eda_df["primary_genre"].value_counts().head(8).index
        top_primary_df = eda_df[eda_df["primary_genre"].isin(top_primary_genres)]

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=top_primary_df,
            x="primary_genre",
            y="vote_average",
            hue="primary_genre",
            palette="pastel",
            legend=False,
        )
        plt.title("Vote Average by Top Primary Genres")
        plt.xlabel("Primary Genre")
        plt.ylabel("Vote Average")
        plt.xticks(rotation=30)
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Observation:** Ratings vary by genre. Some genres have a wider spread, which means audience response is less consistent.
        """
    ),
    markdown_cell("## 10. Correlation Analysis"),
    code_cell(
        """
        # Select numerical columns for correlation analysis
        correlation_columns = [
            "popularity",
            "vote_count",
            "vote_average",
            "release_year",
            "release_month",
            "title_length",
            "overview_length",
        ]

        correlation_matrix = eda_df[correlation_columns].corr()

        print("Correlation matrix:")
        display(correlation_matrix)

        # Heatmap: correlation analysis
        plt.figure(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()
        """
    ),
    markdown_cell(
        """
        **Correlation notes:**

        - A positive correlation means two variables move in the same direction.
        - A negative correlation means one variable tends to increase when the other decreases.
        - A value close to 1 or -1 shows a strong relationship.
        - A value close to 0 shows a weak relationship.

        Most relationships are weak. Popularity is more connected with vote count than with vote average.
        """
    ),
    markdown_cell("## 11. Key Insights"),
    code_cell(
        """
        # Generate simple insight values from the cleaned data
        most_common_genre = all_genres.value_counts().idxmax()
        most_common_genre_count = all_genres.value_counts().max()

        most_common_language = eda_df["original_language"].value_counts().idxmax()
        most_common_language_count = eda_df["original_language"].value_counts().max()

        peak_year = yearly_movies.loc[yearly_movies["movie_count"].idxmax(), "release_year"]
        peak_year_count = yearly_movies["movie_count"].max()

        popularity_rating_corr = eda_df["popularity"].corr(eda_df["vote_average"])
        popularity_vote_count_corr = eda_df["popularity"].corr(eda_df["vote_count"])

        top_popular_movies = movies.sort_values("popularity", ascending=False)[
            ["title", "popularity", "vote_average", "vote_count", "genre"]
        ].head(5)

        qualified_top_rated = movies[movies["vote_count"] >= 500].sort_values("vote_average", ascending=False)[
            ["title", "vote_average", "vote_count", "genre"]
        ].head(5)

        print("Most common genre:", most_common_genre, "with", most_common_genre_count, "movies")
        print("Most common original language:", most_common_language, "with", most_common_language_count, "movies")
        print("Year with the most movie releases:", int(peak_year), "with", int(peak_year_count), "movies")
        print("Correlation between popularity and vote average:", round(popularity_rating_corr, 3))
        print("Correlation between popularity and vote count:", round(popularity_vote_count_corr, 3))

        print("\\nTop 5 movies by popularity:")
        display(top_popular_movies)

        print("Top 5 highly rated movies with at least 500 votes:")
        display(qualified_top_rated)
        """
    ),
    markdown_cell(
        """
        **Key insights:**

        - Drama is one of the most common genres in the dataset.
        - English is the most common original language.
        - Movie releases increased strongly in recent decades, with a visible rise after the 2000s.
        - Popularity and vote average have a weak relationship, so a popular movie is not always highly rated.
        - Vote count has a stronger relationship with popularity than vote average does.
        - Action, animation, and science fiction related movies tend to perform well in average popularity.
        - Some older classic movies have very high ratings when we filter for movies with a reasonable number of votes.
        """
    ),
    markdown_cell("## 12. Conclusion"),
    markdown_cell(
        """
        In this project, I completed an Exploratory Data Analysis of the MovieDB dataset using Python, Pandas, NumPy, Matplotlib, and Seaborn.

        I loaded the data, checked its structure, cleaned missing and invalid values, removed duplicates, converted columns to the correct data types, handled outliers, and created simple features such as release year, release month, title length, and overview length.

        The analysis showed that popularity is unevenly distributed, English-language movies dominate the dataset, and genres such as Drama, Comedy, Action, and Thriller appear frequently. Movie releases also increased in recent decades.

        Overall, this project shows basic data cleaning, preprocessing, visualization, correlation analysis, and insight generation skills in a clear internship-ready format.
        """
    ),
]


notebook = nbf.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
)


with NOTEBOOK_PATH.open("w", encoding="utf-8") as file:
    nbformat.write(notebook, file)


client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(PROJECT_DIR)}},
)
client.execute()


with NOTEBOOK_PATH.open("w", encoding="utf-8") as file:
    nbformat.write(notebook, file)


print(f"Notebook created and executed: {NOTEBOOK_PATH}")
