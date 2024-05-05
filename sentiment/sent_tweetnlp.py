import os
import pandas as pd
import torch
import tweetnlp

VERSION = "tweetnlp3"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("you are using ", device)

print("======================================== testing model ========================================")
model_sent = tweetnlp.load_model('sentiment')
model_irony = tweetnlp.load_model('irony')
model_emotion = tweetnlp.load_model('emotion', multi_label=True)
print("sentiment")
# print(model_sent.sentiment("Yes, including Medicare and social security saving👍", return_probability=True))
print("irony")
print(model_irony.irony("Yes, including Medicare and social security saving👍", return_probability=True))
print("emotion")
print(model_emotion.emotion("Yes, including Medicare and social security saving👍", return_probability=True))
print("=================================== test complete ========================================")
reader = pd.read_csv('data/Bitcoin_tweets_final.csv', chunksize=10000, lineterminator="\n")
i = 1
processed = 0
skipped = 0
error = 0
total = 486
chunks = []

save_dir = f'data/{VERSION}'
# check if directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for chunk in reader:
    save_name = f'data/{VERSION}/Bitcoin_tweets_final_sentiment_v{VERSION}_chunk_{i}.pkl'
    try:
        print(f"processing chunk {i}/{total}...")
        # check if file already exists
        if os.path.isfile(save_name):
            print(f"chunk {i} already processed, skipping...")
            skipped += 1
            continue

        # notStr = chunk['text'].apply(lambda x: type(x) != str)
        # if sum(notStr) > 0:
        #     print(f"chunk {i} has {sum(notStr)} non String entries")

        # sentiment
        chunk['sentiment'] = chunk['text'].apply(lambda x: model_sent.sentiment(x, return_probability=True))
        print(chunk['sentiment'])
        # chunk['sentiment_label'] = chunk['sentiment'].apply(lambda x: x["label"])
        chunk['negative_prob'] = chunk['sentiment'].apply(lambda x: x["probability"]["negative"])
        chunk['neutral_prob'] = chunk['sentiment'].apply(lambda x: x["probability"]["neutral"])
        chunk['positive_prob'] = chunk['sentiment'].apply(lambda x: x["probability"]["positive"])
        chunk = chunk.drop(columns=['sentiment'])

        # irony
        chunk['irony'] = chunk['text'].apply(lambda x: model_irony.irony(x, return_probability=True))
        print(chunk['irony'])
        # chunk['irony_label'] = chunk['irony'].apply(lambda x: x["label"])
        chunk['irony_prob'] = chunk['irony'].apply(lambda x: x["probability"]["irony"])
        chunk = chunk.drop(columns=['irony'])

        # emotion
        chunk['emotion'] = chunk['text'].apply(lambda x: model_emotion.emotion(x, return_probability=True))
        print(chunk['emotion'])
        # chunk['emotion_label'] = chunk['emotion'].apply(lambda x: x["label"])
        chunk['anger_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["anger"])
        chunk['anticipation_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["anticipation"])
        chunk['disgust_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["disgust"])
        chunk['fear_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["fear"])
        chunk['joy_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["joy"])
        chunk["love_prob"] = chunk['emotion'].apply(lambda x: x["probability"]["love"])
        chunk['optimism_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["optimism"])
        chunk['pessimism_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["pessimism"])
        chunk['sadness_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["sadness"])
        chunk['surprise_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["surprise"])
        chunk['trust_prob'] = chunk['emotion'].apply(lambda x: x["probability"]["trust"])
        chunk = chunk.drop(columns=['emotion'])
        # chunks.append(chunk)
        chunk.to_pickle(save_name)

        processed += 1

    except Exception as e:
        print(e)
        print(f"ERROR error processing chunk {i}")
        skipped += 1
        error += 1
        continue
    finally:
        i += 1
        print(f"processed {processed} chunks")
        print(f"skipped {skipped} chunks")
        print(f"{error} chunks had errors")

# print("done processing all chunks")
# print("writing to disk as pickle file...")

# df = pd.concat(chunks)
# df.to_pickle(f'data/Bitcoin_tweets_final_sentiment_v{VERSION}.pkl')
print(f"processed {processed} chunks")
print(f"skipped {skipped} chunks")
print(f"{error} chunks had errors")
print("done")
