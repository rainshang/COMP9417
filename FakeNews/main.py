import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from textblob import TextBlob
from scipy.spatial.distance import cosine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

__MAX_FEATURES = 5000
__TRAINED_DATA_SET = 'dataset.csv'


def train():
    stances_train = pd.read_csv('data/train_stances.csv')
    bodies_train = pd.read_csv('data/train_bodies.csv')

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
    # Headline | tf_headline | tf_idf_headline | polarity_headline | subjectivity_headline
    # articleBody | tf_body | tf_idf_body | polarity_body | subjectivity_body
    headline_df = pd.DataFrame(headlines, columns=['Headline'])
    body_df = pd.DataFrame(bodies, columns=['articleBody'])

    headline_df['tf_headline'] = pd.DataFrame(
        tf[:len(headlines)], columns=feature_names).values.tolist()
    headline_df['tf_idf_headline'] = pd.DataFrame(
        tf_idf[:len(headlines)], columns=feature_names).values.tolist()

    body_df['tf_body'] = pd.DataFrame(
        tf[len(headlines):len(vocabulary)], columns=feature_names).values.tolist()
    body_df['tf_idf_body'] = pd.DataFrame(
        tf_idf[len(headlines):len(vocabulary)], columns=feature_names).values.tolist()

    # left join
    headline_df = stances_train.merge(headline_df, on='Headline')
    body_df = bodies_train.merge(body_df, on='articleBody')

    for i in range(headline_df.shape[0]):
        blob = TextBlob(headline_df['Headline'][i])
        headline_df.loc[i, 'polarity_headline'] = blob.sentiment.polarity
        headline_df.loc[i,
                        'subjectivity_headline'] = blob.sentiment.subjectivity

    for i in range(body_df.shape[0]):
        blob = TextBlob(body_df['articleBody'][i])
        body_df.loc[i, 'polarity_body'] = blob.sentiment.polarity
        body_df.loc[i, 'subjectivity_body'] = blob.sentiment.subjectivity

    # ('Headline', '_1', 'Stance', 'tf_headline', 'tf_idf_headline', 'polarity_headline', 'subjectivity_headline', 'articleBody', 'tf_body', 'tf_idf_body', 'polarity_body', 'subjectivity_body')
    df = headline_df.merge(body_df, on='Body ID')

    similarity = []
    for entry in df.itertuples(index=False):
        # cosine similarity of tf_idf_headline and tf_idf_body
        if sum(entry[4]) == 0 or sum(entry[9]) == 0:
            similarity.append(0)
        else:
            similarity.append(1 - cosine(entry[4], entry[9]))
    df['tf_idf_cos_sim'] = pd.Series(similarity)

    # flat the TF to features, df will be like
    # tf_headline | tf_body | tf_idf_cos_sim | relatedness | tf_headline in feature_names | tf_body in feature_names
    tf_headline = pd.DataFrame(
        df['tf_headline'].tolist(), columns=feature_names)
    tf_body = pd.DataFrame(df['tf_body'].tolist(), columns=feature_names)

    df = pd.concat([df, tf_headline], axis=1)
    df = pd.concat([df, tf_body], axis=1)

    df = df.drop(['Headline', 'Body ID', 'tf_headline', 'tf_idf_headline',
                  'articleBody', 'tf_body', 'tf_idf_body'], axis=1)
    print(df)

    df.to_csv('dataset.csv')
    return df


def main():
    if os.path.exists(__TRAINED_DATA_SET):
        df = pd.read_csv(__TRAINED_DATA_SET)
    else:
        df = train()

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # Encoding the Dependent Variable
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    lda = LDA(n_components=None)
    x_train = lda.fit_transform(x_train, y_train)
    x_test = lda.transform(x_test)
    explained_variance = lda.explained_variance_ratio_

    # Fitting Logistic Regression to the Training set
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    # Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    # Fitting SVM to the Training set
    classifier = SVC(kernel='linear', random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    # Fitting Kernel SVM to the Training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    # Fitting Decision Tree Classification to the Training set
    classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(
        n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))

    parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1)
    grid_search = grid_search.fit(x_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    # Fitting Kernel SVM to the Training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    (cm[0][0] + cm[1][1] + cm[2][2] + cm[3][3]) / sum(sum(cm))


if __name__ == '__main__':
    main()
