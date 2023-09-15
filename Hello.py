import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file into a DataFrame
df = pd.read_csv("spotify_millsongdata.csv")

# Display the first 10 and last 10 rows of the DataFrame
print(df.head(10))
print(df.tail(10))

# Sample and preprocess the DataFrame
df = df.sample(5000).drop('link', axis=1).reset_index(drop=True)
df = df.sample(5000)
df['text'] = df['text'].str.lower().replace(r'^\w\s', '').replace(r'\n', '', regex=True)

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Define a function to tokenize and stem the text
def token(txt):
    tokens = nltk.word_tokenize(txt)
    stemmed_tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(stemmed_tokens)

# Apply the tokenization and stemming to the 'text' column
df['text'] = df['text'].apply(lambda x: token(x))

# Create TF-IDF vectors
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidf.fit_transform(df['text'])
similar = cosine_similarity(matrix)

# Define a function to recommend songs based on similarity to a given song
def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similar[idx])), reverse=True, key=lambda x: x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]]['song'])
        
    return songs

# Example of recommending songs similar to 'Someone Like You'
recommended_songs = recommendation('Someone Like You')
print("Recommended Songs:")
for i, song in enumerate(recommended_songs, start=1):
    print(f"{i}. {song}")
