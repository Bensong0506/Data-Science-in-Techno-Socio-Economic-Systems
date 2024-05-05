import pandas as pd
import pickle

PICKLE_DIR = 'data/tweetnlp3/'
PICKLE_NAME = lambda i: f"Bitcoin_tweets_final_sentiment_vtweetnlp3_chunk_{i}.pkl"
OUTPUT_NAME = 'tweets.pkl'

print(f"your pickle is compatible with {pickle.compatible_formats}")

print("loading chunks...")
# open the pickle chunks and concatenate them
chunks = [pd.read_pickle(PICKLE_DIR + PICKLE_NAME(i)) for i in range(1, 486)]
print(f"{len(chunks)} chunks loaded...")
df = pd.concat(chunks)

print("writing to pickle file...")
# write to a new pickle file
df.to_pickle(PICKLE_DIR + OUTPUT_NAME)
print("done")
