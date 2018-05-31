# -*- coding: utf-8 -*-
"""
Created on Thu May 31 16:35:57 2018

@author: 40227
"""

import pandas as pd
import numpy as np

def mean_squared_error(pred, true):
    return np.average((pred - true) ** 2)

def cos_similarity(Rating_matrix):
    similarity = np.dot(Rating_matrix, Rating_matrix.T) + 1e-9
    norms = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / (norms * norms.T))

def recommand_from_other_users(k, uid, Rating_matrix):
    user_index = uid - 1
    num_other_user = 50
    #user cos similarity matrix:
    user_sim_matrix = cos_similarity(Rating_matrix)
    #empty prediction matrix:   
    pred_matrix = np.zeros(Rating_matrix.shape)
    # exclude the get the top-50 users' indexes other than user itself        
    index_of_top_50 = [np.argsort(user_sim_matrix[:,user_index])[-2:-num_other_user-2:-1]]

    for item in range(Rating_matrix.shape[1]):
        if Rating_matrix[user_index][item] == 0:
            # Denominator is the sum of similarity for each user with its top 50 users:
            denominator = np.sum(user_sim_matrix[user_index,:][index_of_top_50])
            
            # Numerator
            numerator = user_sim_matrix[user_index,:][index_of_top_50].dot(Rating_matrix[:,item][index_of_top_50])
            
            pred_matrix[user_index, item] = numerator/denominator
                
            #print('Top-' + str(k) + ': Prediction for user ' + str(user) + '/' + str(training_set.shape[0]) + ' done...')
    movie_ids = [i for i in np.argsort(pred_matrix[user_index, :])[-k:]]
    # return the movie id that this user has not rated
    # but his Top-50 similar user rate it high
    return movie_ids

def read_data(file):
    # read data from file
    Ratings_Names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
    raw_df = pd.read_csv(file, skiprows=1, sep='\t', names=Ratings_Names)
    
    #number of users and movies
    num_of_users = max(raw_df['User_ID'])
    num_of_movies = max(raw_df['Movie_ID'])
    
    #empty Rating_matrix
    Rating_matrix = np.zeros((num_of_users, num_of_movies))
    #built Rating_matrix by raw_df
    for line in raw_df.itertuples():
        Rating_matrix[line[1] - 1][line[2] - 1] = line[3]
    #sparsity of the matrix
    matrix_sparsity = len(raw_df) / (num_of_users * num_of_movies)
    
    return raw_df, Rating_matrix, matrix_sparsity

raw_df, Rating_matrix, matrix_sparsity = read_data('u.data')

recommand_from_other_users(10, 1, Rating_matrix)



















