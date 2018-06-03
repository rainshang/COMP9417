import sys
import pandas
import numpy
import matplotlib.pyplot as pyplot

DEBUG = False

global user_list
global user_id_index_map
global movie_list
global rating_matrix
global rating_sim_matrix
global movies


def main():
    if len(sys.argv) == 1:
        rating_data_path = 'ml-latest-small/ratings.csv'
        movie_data_path = 'ml-latest-small/movies.csv'
    else:
        rating_data_path = sys.argv[1]
        movie_data_path = sys.argv[2]

    ratings = parse_data(rating_data_path)
    rating_matrix_sparsity = generate_rating_matrix(ratings)
    if DEBUG:
        evaluation(rating_matrix_sparsity)

    global rating_sim_matrix
    rating_sim_matrix = cos_similarity(rating_matrix)
    global movies
    movies = parse_data(movie_data_path)

    print('=================================================================================\n'
          'Welecome to Group BIG BUN\'s Collaborative Filtering based Movie Recommation System.\n'
          '                                         by Ang Li\n'
          '                                            Ethan Xu\n'
          '                                            Ge Huang\n'
          '                                            Zhizheng Shi\n\n'
          '  1.Input the ID of user to get recommandaton.\n'
          '  2.Input \'q\' or other non-integer to exit.\n\n'
          '=================================================================================')

    while True:
        inputs = input('Please input the user ID:')
        try:
            uid = int(inputs)
            if uid in user_list:
                movie_ids = recommend_movies(5, 50, uid)
                print(
                    '\nAccording to these users\'s preference, we recommand you 5 movies:\n')
                print('{}\n'.format(
                    movies.loc[movies[movies.columns[0]].isin(movie_ids)]))
            else:
                print('Opps, user-{} seems not existing...'.format(uid))
        except ValueError:
            exit(0)


def parse_data(path):
    if path.endswith('.csv'):
        return pandas.read_csv(path)
    # elif path.endswith('.data'):
    #     return pandas.read_csv(path, sep='\t')
    else:
        raise TypeError('We don\'t support other data format!')


def generate_rating_matrix(ratings):
    global user_list
    global user_id_index_map
    global movie_list
    global rating_matrix
    # get all users and movies in rating data
    user_set = set()
    movie_set = set()
    for entry in ratings.itertuples(index=False):
        user_id = entry[0]
        movie_id = entry[1]
        user_set.add(user_id)
        movie_set.add(movie_id)
    user_num = len(user_set)
    item_num = len(movie_set)
    if DEBUG:
        print('{} users and {} movies involved'.format(user_num, item_num))

    # sort the id, this step is not necessary
    user_list = sorted(user_set)
    movie_list = sorted(movie_set)

    # create a hash map for searching the position in user_list(movie_list) of one user_id(movie_id)
    user_id_index_map = {}
    movie_id_index_map = {}
    for i in range(user_num):
        user_id_index_map[user_list[i]] = i
    for i in range(item_num):
        movie_id_index_map[movie_list[i]] = i

    # create rating matrix
    rating_matrix = numpy.zeros((user_num, item_num))
    rating_matrix_sparsity = len(ratings) / (user_num * item_num)
    if DEBUG:
        print('Sparsity of rating matrix is: {}'.format(rating_matrix_sparsity))
    for entry in ratings.itertuples(index=False):
        user_id = entry[0]
        movie_id = entry[1]
        index = user_id_index_map[user_id]
        column = movie_id_index_map[movie_id]
        rating_matrix[index, column] = entry[2]

    return rating_matrix_sparsity


def evaluation(rating_matrix_sparsity):
    # we select 10% of sparsity * item as testing data set for each user
    num_test_items_per_user = int(
        0.1 * rating_matrix_sparsity * rating_matrix.shape[1])
    print('Select {} movies for testing per user'.format(
        num_test_items_per_user))

    train_data = rating_matrix.copy()
    test_data = numpy.zeros(rating_matrix.shape)
    for user_index in range(rating_matrix.shape[0]):
        # randomly select (num_test_items_per_user) different rated movies from each user
        movie_index = numpy.random.choice(rating_matrix[user_index].nonzero()[
            0], size=num_test_items_per_user, replace=False)
        test_data[user_index, movie_index] = rating_matrix[user_index, movie_index]
        train_data[user_index, movie_index] = 0.

    # calculate cosine-similarity
    train_sim_matrix = cos_similarity(train_data)

    # predict based on Top-k based CF and normal CF
    performance = []
    ks = [5, 10, 50, 100, 200, 'ALL']
    for k in ks:
        performance.append(predict_based_on_users(
            k, train_sim_matrix, train_data, test_data))

    # visualization
    y_pos = numpy.arange(len(ks))
    pyplot.bar(y_pos, performance, align='center', alpha=0.5)
    pyplot.xticks(y_pos, ks)
    pyplot.ylabel('MSE')
    pyplot.title('MSE of different k values')
    pyplot.show()


def cos_similarity(matrix):
    similarity = numpy.dot(matrix, matrix.T) + 1e-9
    norms = numpy.array([numpy.sqrt(numpy.diagonal(similarity))])
    return (similarity / (norms * norms.T))


def predict_based_on_users(k, user_sim_matrix, train_data, test_data):
    if k == 'ALL':
        prediction_matrix = user_sim_matrix.dot(
            train_data) / numpy.array([numpy.abs(user_sim_matrix).sum(axis=1)]).T
    else:
        num_of_users = train_data.shape[0]
        num_of_movies = train_data.shape[1]
        # empty prediction matrix:
        prediction_matrix = numpy.zeros((num_of_users, num_of_movies))
        for user in range(num_of_users):
            # indexes of k most similar users
            index_k_most_similar_users = [numpy.argsort(
                user_sim_matrix[:, user])[-2:-k-2:-1]]
            for movie in range(num_of_movies):
                denominator = numpy.sum(
                    user_sim_matrix[user][index_k_most_similar_users])
                numerator = user_sim_matrix[user][index_k_most_similar_users].dot(
                    train_data[:, movie][index_k_most_similar_users])
                prediction_matrix[user][movie] = numerator / denominator
    # index of non-zero values of test_data
    non_zero_index = test_data.nonzero()
    # mse of test data and predict data
    mse = MSE(prediction_matrix[non_zero_index].flatten(
    ), test_data[non_zero_index].flatten())
    print('The mean squared error of {} user_based CF is: {}\n'.format(k, mse))
    return mse


def MSE(pred, true):
    return numpy.average((pred - true) ** 2)


def recommend_movies(n, k, uid):
    user_index = user_id_index_map[uid]
    # indexes of (num_other_user) most similar users
    index_most_similar_users = numpy.argsort(
        rating_sim_matrix[:, user_index])[-2:-k-2:-1]
    print('We\'ve found {} users whose tastes are most similar with you'.format(
        len(index_most_similar_users)))

    num_of_movies = rating_matrix.shape[1]
    # empty prediction matrix:
    pred_matrix = numpy.zeros(num_of_movies)
    for movie in range(num_of_movies):
        if rating_matrix[user_index][movie] == 0:
            denominator = numpy.sum(
                rating_sim_matrix[user_index][index_most_similar_users])
            numerator = rating_sim_matrix[user_index][index_most_similar_users].dot(
                rating_matrix[:, movie][index_most_similar_users])
            pred_matrix[movie] = numerator / denominator

    # unrated movies of this user but hightly rated by most similar users
    # actual movie id = movie index + 1
    movie_ids = [movie_list[i] for i in numpy.argsort(pred_matrix)[-n:]]
    return movie_ids


if __name__ == '__main__':
    main()
