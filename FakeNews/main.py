import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from scipy.spatial.distance import cosine


__MAX_FEATURES = 5000


def main():
    stances_train = pd.read_csv('data/train_stances.csv', encoding='utf-8')
    bodies_train = pd.read_csv('data/train_bodies.csv', encoding='utf-8')

    # arrays of headline and body
    headlines = [headline for headline in stances_train['Headline'].unique()]
    bodies = [body for body in bodies_train['articleBody'].unique()]

    # calculate the BOW, TF and TF-IDF of training data
    vocabulary = headlines + bodies

    bow_vectorizer = CountVectorizer(
        max_features=__MAX_FEATURES, stop_words=ENGLISH_STOP_WORDS)
    bow = bow_vectorizer.fit_transform(vocabulary)
    feature_names = bow_vectorizer.get_feature_names()

    tf = TfidfTransformer(use_idf=False).fit_transform(bow).toarray()
    tf_idf = TfidfTransformer().fit_transform(bow).toarray()

    # 2 new tables:
    # Headline | tf | tf_idf     and     articleBody | tf | tf_idf
    headline_df = pd.DataFrame(headlines, columns=['Headline'])
    body_df = pd.DataFrame(bodies, columns=['articleBody'])

    headline_df['tf'] = pd.DataFrame(
        tf[:len(headlines)], columns=feature_names).values.tolist()
    headline_df['tf_idf'] = pd.DataFrame(
        tf_idf[:len(headlines)], columns=feature_names).values.tolist()

    body_df['tf'] = pd.DataFrame(
        tf[len(headlines):len(vocabulary)], columns=feature_names).values.tolist()
    body_df['tf_idf'] = pd.DataFrame(
        tf_idf[len(headlines):len(vocabulary)], columns=feature_names).values.tolist()

    # left join
    headline_df = stances_train.merge(headline_df, on='Headline')
    body_df = bodies_train.merge(body_df, on='articleBody')

    # Headline | vector_headline | articlebody | vector_body | relatedness
    df = headline_df.merge(body_df, on='Body ID').rename(
        columns={'tf_x': 'tf_headline', 'tf_y': 'tf_body', 'tf_idf_x': 'tf_idf_headline', 'tf_idf_y': 'tf_idf_body'})

    df.apply(lambda row: 'unrelated' if row['Stance'] ==
             'unrelated' else 'related', axis=1).value_counts()

    df['relatedness'] = df.apply(
        lambda row: 1 if row['Stance'] != 'unrelated' else 0, axis=1)

    similarity = []
    for entry in df.itertuples(index=False):
        # cosine similarity of tf_idf_headline and tf_idf_body
        similarity.append(1 - cosine(entry[4], entry[7]))

    df['tf_idf_cos_sim'] = pd.Series(similarity)

    df = df[['tf_headline', 'tf_body', 'tf_idf_cos_sim', 'relatedness']]

    print(df)


if __name__ == '__main__':
    main()
