import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

# Load the character data
with open('reviews.json', 'r', encoding='utf-8') as f:
    characters = json.load(f)

# Load the personality traits
with open('chineseTraits.json', 'r', encoding='utf-8') as f:
    traits = json.load(f)

# Initialize a CountVectorizer to count the occurrences of traits
vectorizer = CountVectorizer(vocabulary=traits)

# Create a dictionary to hold the data
data = defaultdict(list)

# Iterate over the characters
for character in characters:
    name = character['Name']
    reviews = character['Reviews']
    # Use the CountVectorizer to count the occurrences of traits
    X = vectorizer.fit_transform(reviews)
    # The output is a sparse matrix, convert it to an array and sum along the columns
    counts = X.toarray().sum(axis=0)
    # Append the counts to the data
    for trait, count in zip(traits, counts):
        data[trait].append(count)

# Convert the data to a DataFrame
df = pd.DataFrame(data, index=[character['Name'] for character in characters])

# Replace NaNs with 0s
df.fillna(0, inplace=True)

# Calculate the total count of each trait for each character
df['Total'] = df.iloc[:, 1:].sum(axis=1) 

# Calculate the frequency of each trait for each character
for col in df.columns[1:-1]: 
    df[col] = df[col] / df['Total']

# Map the frequency to a range of 0-5
for col in df.columns[1:-1]:
    df[col] = pd.cut(df[col], bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf], labels=[0, 1, 2, 3, 4, 5])

# Drop the 'Total' column as we no longer need it
df.drop(columns=['Total'], inplace=True)

# Display the updated DataFrame
print(df)