# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:35:57 2018

@author: 40227
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def MSE(pred, true):
    return np.average((pred - true) ** 2)

def cos_similarity(Rating_matrix):
    similarity = np.dot(Rating_matrix, Rating_matrix.T) + 1e-9
    norms = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / (norms * norms.T))

def recommend_movies(n, k, uid, rating_matrix):
    #actual index of user is uid - 1
    user_index = uid - 1
    #user cos similarity matrix:
    user_sim_matrix = cos_similarity(rating_matrix)
    #indexes of (num_other_user) most similar users 
    index_most_similar_users = [np.argsort(user_sim_matrix[:,user_index])[-2:-k-2:-1]]
    
    num_of_movies = rating_matrix.shape[1]
    #empty prediction matrix:   
    pred_matrix = np.zeros(num_of_movies)
    for movie in range(num_of_movies):
        if rating_matrix[user_index][movie] == 0:
            denominator = np.sum(user_sim_matrix[user_index][index_most_similar_users])
            numerator = user_sim_matrix[user_index][index_most_similar_users].dot(rating_matrix[:,movie][index_most_similar_users])
            pred_matrix[movie] = numerator / denominator
    #unrated movies of this user but hightly rated by most similar users
    #actual movie id = movie index + 1
    movie_ids = [i + 1 for i in np.argsort(pred_matrix)[-n:]]
    return movie_ids

def predict_based_on_users (k, user_sim_matrix, train_data, test_data):
    if k == 'ALL':
        prediction_matrix = user_sim_matrix.dot(train_data) / np.array([np.abs(user_sim_matrix).sum(axis=1)]).T
    else:
        num_of_users = train_data.shape[0]
        num_of_movies = train_data.shape[1]
        #empty prediction matrix:   
        prediction_matrix = np.zeros((num_of_users, num_of_movies))
        for user in range(num_of_users):
            #indexes of k most similar users  
            index_k_most_similar_users = [np.argsort(user_sim_matrix[:,user])[-2:-k-2:-1]]
            for movie in range(num_of_movies):
                denominator = np.sum(user_sim_matrix[user][index_k_most_similar_users])
                numerator = user_sim_matrix[user][index_k_most_similar_users].dot(train_data[:,movie][index_k_most_similar_users])
                prediction_matrix[user][movie] = numerator / denominator
    #index of non-zero values of test_data
    non_zero_index = test_data.nonzero()
    #mse of test data and predict data
    mse = MSE(prediction_matrix[non_zero_index].flatten(), test_data[non_zero_index].flatten())
    print('The mean squared error of {} user_based CF is: {}\n'.format(k, mse))
    return mse

def evaluation(raw_df, Rating_matrix, shape):
    #number of data lines
    num_lines = len(raw_df)
    #built Rating_matrix by raw_df
    for line in raw_df.itertuples():
        Rating_matrix[line[1] - 1][line[2] - 1] = line[3]
    #sparsity of the matrix
    matrix_sparsity = num_lines / (shape[0] * shape[1])
    
    num_ratings = int(0.1 * matrix_sparsity * shape[1])
    print('select {} ratings from each user'.format(num_ratings))
    
    train_data = Rating_matrix.copy()
    test_data = np.zeros(shape)
    #actual user id = user_index + 1
    for user_index in range(shape[0]):
        #randomly select (num_ratings) different rated movies from each user
        movie_index = np.random.choice(Rating_matrix[user_index].nonzero()[0], size = num_ratings, replace = False)
        #take ratings of these movies into test_data
        test_data[user_index, movie_index] = Rating_matrix[user_index, movie_index]
        #set same position of train_data to 0
        train_data[user_index, movie_index] = 0.

    user_sim_matrix = cos_similarity(train_data)
    performance = []
    ks = [5, 10, 50, 100, 200, 'ALL']
    for k in ks:
        performance.append(predict_based_on_users (k, user_sim_matrix, train_data, test_data))
    y_pos = np.arange(len(ks)) 
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, ks)
    plt.ylabel('MSE')
    plt.title('MSE of different k values')
    plt.show()

def read_rating_data(file):
    # read data from file
    Ratings_Names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
    raw_df = pd.read_csv(file, skiprows=1, sep='\t', names=Ratings_Names)
#    raw_df = pd.read_csv(file)
    columns = raw_df.columns
    #number of users and movies
    shape = (max(raw_df[columns[0]]), max(raw_df[columns[1]]))
    #empty Rating_matrix
    rating_matrix = np.zeros(shape)
    return raw_df, rating_matrix, shape

def read_user_data(file):
    # read data from file
    raw_df = pd.read_csv(file)
    columns = raw_df.columns
    
def read_movie_data(file, movie_ids):
    # read data from file
    raw_df = pd.read_csv(file)
    columns = raw_df.columns
    print(raw_df.loc[raw_df[columns[0]].isin(movie_ids)][columns[1]])
    return raw_df.loc[raw_df[columns[0]].isin(movie_ids)][columns[1]]
    
def read_info_data(file):
    # read data from file
    raw_df = pd.read_csv(file)
    columns = raw_df.columns

rating_df, rating_matrix, shape = read_rating_data('ml-100k/u.data')
evaluation(rating_df, rating_matrix, shape)
movie_ids = recommend_movies(5, 50, 1, rating_matrix)
#read_movie_data('movies.csv', movie_ids)






















