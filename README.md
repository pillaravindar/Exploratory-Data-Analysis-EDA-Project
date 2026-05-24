# Exploratory-Data-Analysis-EDA-Project
# MovieDB Exploratory Data Analysis

## Project Overview

This project performs Exploratory Data Analysis on a MovieDB dataset. The main aim is to understand movie trends based on popularity, ratings, vote counts, genres, languages, and release dates.

The analysis is written in a beginner-friendly way using Python and common data analysis libraries.

## Files Included

- `MovieDB_EDA_Intern_Project.ipynb` - Main Jupyter Notebook containing the full EDA project.
- `mymoviedb.csv` - Dataset used for the analysis.

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

## Project Sections

The notebook is organized into the following sections:

1. Project Introduction
2. Problem Statement
3. Import Libraries
4. Load Dataset
5. Dataset Overview
6. Data Cleaning
7. Feature Engineering
8. Exploratory Data Analysis
9. Visualizations
10. Correlation Analysis
11. Key Insights
12. Conclusion

## Main Work Done

- Loaded and inspected the dataset.
- Checked dataset shape, columns, data types, and summary statistics.
- Cleaned missing and invalid values.
- Removed duplicate rows.
- Converted date and numeric columns to proper data types.
- Detected and handled outliers using the IQR method.
- Created new features such as release year, release month, title length, and overview length.
- Analyzed popularity, vote average, vote count, genre, and original language.
- Created visualizations including histograms, bar charts, countplots, scatter plots, line charts, boxplots, and a correlation heatmap.
- Summarized key insights and final conclusions.

## How to Run

1. Open `MovieDB_EDA_Intern_Project.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. Make sure `mymoviedb.csv` is in the same folder as the notebook.
3. Run the notebook cells from top to bottom.

## Key Insights

- Drama, Comedy, Action, and Thriller are among the most common genres.
- English is the most common original language in the dataset.
- Movie releases increased noticeably from the 2000s onward.
- Popularity and vote average have a weak relationship.
- Vote count is more closely related to popularity than vote average.
- Action, Animation, and Science Fiction movies show strong average popularity.

## Conclusion

This project shows the complete EDA workflow for a movie dataset. It covers data loading, cleaning, preprocessing, feature engineering, visualization, correlation analysis, and insight generation in a clear and simple format.
