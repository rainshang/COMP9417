import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial


def main():
    stances_train = pd.read_csv('data/train_stances.csv', encoding='utf-8')
    bodies_train = pd.read_csv('data/train_bodies.csv', encoding='utf-8')

    # arrays of headline and body
    headlines = [headline for headline in stances_train["Headline"].unique()]
    bodies = [body for body in bodies_train["articleBody"].unique()]

    # calculate term frequency
    vocabulary = headlines + bodies
    count_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    term_frequencies = count_vectorizer.fit_transform(vocabulary).toarray()
    feature_names = count_vectorizer.get_feature_names()

    # 2 new tables:
    # headline | vector     and     body | vector
    headline_df = pd.DataFrame(headlines, columns=["Headline"])
    body_df = pd.DataFrame(bodies, columns=["articleBody"])

    headline_df["vector"] = pd.DataFrame(
        term_frequencies[:len(headlines)], columns=feature_names).values.tolist()
    body_df["vector"] = pd.DataFrame(
        term_frequencies[len(headlines):len(vocabulary)], columns=feature_names).values.tolist()

    headline_df = stances_train.merge(headline_df, on='Headline')
    body_df = bodies_train.merge(body_df, on='articleBody')

    df = headline_df.merge(body_df, on='Body ID').rename(
        columns={'vector_x': 'vector_headline', 'vector_y': 'vector_body'})

    df.apply(lambda row: 'unrelated' if row['Stance'] ==
             'unrelated' else 'related', axis=1).value_counts()

    df['relatedness'] = df.apply(
        lambda row: 1 if row['Stance'] != 'unrelated' else 0, axis=1)

    df = df.drop(['Stance', 'Body ID'], axis=1)

    similarity = []
    for i in range(49971):
        # a = cosine_similarity([df["vector_headline"].loc[i]], [
        #                       df["vector_body"].loc[i]])[0][0]
        b = 1 - \
            spatial.distance.cosine(
                df["vector_headline"].loc[i], df["vector_body"].loc[i])
        similarity.append(b)

    df["similarity"] = pd.Series(similarity)


if __name__ == '__main__':
    main()
