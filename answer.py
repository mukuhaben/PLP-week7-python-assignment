# Task 1: Load and Explore the Dataset

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load dataset
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Display first few rows
print("\n🔍 First 5 rows of the dataset:")
print(df.head())

# Check data structure
print("\n🧾 Data types and missing values:")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

# No missing values in this dataset, otherwise you'd do:
# df = df.dropna() or df.fillna(method='ffill')

# Task 2: Basic Data Analysis

# Basic statistics
print("\n📊 Summary statistics:")
print(df.describe())

# Grouping by species and calculating mean of numerical columns
grouped = df.groupby('species').mean()
print("\n📈 Mean values grouped by species:")
print(grouped)

# Example observation
print("\n📌 Observation: Iris-virginica has the highest mean petal length.")

# Task 3: Data Visualization

# Set theme
sns.set(style="whitegrid")

# Line Chart – Not ideal for Iris but using dummy time index
df_with_time = df.copy()
df_with_time['index'] = range(len(df))
plt.figure(figsize=(10, 5))
sns.lineplot(x='index', y='sepal length (cm)', hue='species', data=df_with_time)
plt.title("Line Chart: Sepal Length Trend (Dummy Time Index)")
plt.xlabel("Index (Simulated Time)")
plt.ylabel("Sepal Length (cm)")
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Bar Chart – Average petal length per species
plt.figure(figsize=(7, 5))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram – Distribution of sepal width
plt.figure(figsize=(7, 5))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True)
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Scatter Plot – Sepal Length vs Petal Length
plt.figure(figsize=(7, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.tight_layout()
plt.show()

# Final Findings Summary
print("\n🔎 Final Observations:")
print("- Iris-virginica has the longest petals on average.")
print("- Sepal length and petal length show a positive correlation.")
print("- Histogram shows that most sepal widths are between 2.5 and 3.5 cm.")
print("- Species are distinguishable by petal dimensions in the scatter plot.")
