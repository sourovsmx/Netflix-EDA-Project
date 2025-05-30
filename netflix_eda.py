
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

try:
    df = pd.read_csv("netflix_originals.csv", encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv("netflix_originals.csv", encoding='latin1')
    except Exception as e:
        print(f"Failed to read file: {e}")
        exit()

print("\n=== Initial Data Shape ===")
print(df.shape)

df.dropna(inplace=True)

df['IMDB Score'] = pd.to_numeric(df['IMDB Score'], errors='coerce')
df = df.dropna(subset=['IMDB Score'])

print("\n=== Cleaned Data Shape ===")
print(df.shape)


print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Summary Statistics ===")
print(df.describe())

plt.figure(figsize=(10, 6))
sns.histplot(df['IMDB Score'], bins=20, kde=True, color='red')
plt.title("Distribution of IMDB Scores for Netflix Originals")
plt.xlabel("IMDB Score")
plt.ylabel("Count")
plt.savefig("imdb_distribution.png")
plt.show()


plt.figure(figsize=(12, 6))
top_languages = df['Language'].value_counts().head(10)
sns.barplot(x=top_languages.values, y=top_languages.index, palette="rocket")
plt.title("Top 10 Languages in Netflix Originals")
plt.xlabel("Number of Movies")
plt.ylabel("Language")
plt.savefig("top_languages.png")
plt.show()


fig = px.scatter(df, x='Runtime', y='IMDB Score', 
                 hover_name='Title', color='Genre',
                 title="Runtime vs IMDB Score for Netflix Originals")
fig.show()

fig.write_html("runtime_vs_score.html")

print("\n=== Top 5 Highest Rated Movies ===")
top_movies = df.sort_values('IMDB Score', ascending=False).head(5)
print(top_movies[['Title', 'Genre', 'IMDB Score', 'Premiere']])