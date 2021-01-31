
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # # YOUR CODE HERE FOR Q1.1.1
        # dataset = ratings['item'].iloc[0]
        # maxRatings = 0
        # curRatings = 0
        # for item, rating in zip(ratings.item, ratings.rating):
        #     if (item == dataset):
        #         curRatings = curRatings + rating
        #         dataset = item
        #     else:
        #         if (curRatings > maxRatings):
        #             maxRatings = curRatings
        #             maxRatingItem = dataset
        #         curRatings = 0
        #         curRatings = rating + curRatings
        #         dataset = item
        # print(maxRatings)
        # print(maxRatingItem)
        # n = len(set(ratings["user"]))
        # d = len(set(ratings["item"]))
        #
        # # YOUR CODE HERE FOR Q1.1.2
        # print(ratings.user.mode())
        # print(n)
        # print("\n")
        # print(ratings['user'].value_counts())
        # print(ratings['rating'].value_counts())
        # print(ratings['item'].value_counts())

        # YOUR CODE HERE FOR Q1.1.3
        plt.figure()
        plt.yscale('log', nonposy='clip')
        ratings_per_user = X_binary.getnnz(axis=1)
        plt.hist(ratings_per_user)
        plt.title("ratings_per_user")
        plt.xlabel("number of users")
        plt.ylabel("number of ratings")
        filename = os.path.join("..", "figs", "ratings_per_user")
        plt.savefig(filename)

        plt.figure()
        plt.yscale('log', nonposy='clip')
        ratings_per_item = X_binary.getnnz(axis=0)
        plt.hist(ratings_per_item)
        plt.title("ratings_per_item")
        plt.xlabel("number of items")
        plt.ylabel("number of ratings")
        filename = os.path.join("..", "figs", "ratings_per_item")
        plt.savefig(filename)

        plt.figure()
        plt.hist(ratings['rating'])
        plt.title("Number of Ratings per Rating")
        plt.xlabel("Ratings")
        plt.ylabel("Number of Ratings")
        filename = os.path.join("..", "figs", "ratings")
        plt.savefig(filename)


    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        def get_key(val):
            for key, value in item_mapper.items():
                if val == value:
                    return key
            return "key doesn't exist"

        model = NearestNeighbors(n_neighbors=6)
        model.fit(X.T)
        euclidean = model.kneighbors(grill_brush_vec.T, return_distance=False)
        euclideanItems = []
        for item_ind in euclidean[0]:
            euclideanItems.append(get_key(item_ind))
        print("Euclidean items: ")
        print(euclideanItems)

        model = NearestNeighbors(n_neighbors=6)
        X_normal = normalize(X.T)
        model.fit(X_normal)
        norm_euclidean = model.kneighbors(grill_brush_vec.T, return_distance=False)
        normItems = []
        for item_ind in norm_euclidean[0]:
            normItems.append(get_key(item_ind))
        print("Normalized euclidean items: ")
        print(normItems)

        model = NearestNeighbors(n_neighbors=6, metric='cosine')
        model.fit(X.T)
        cosine = model.kneighbors(grill_brush_vec.T, return_distance=False)
        cosineItems = []
        for item_ind in cosine[0]:
            cosineItems.append(get_key(item_ind))
        print("Cosine items: ")
        print(cosineItems)


        # YOUR CODE HERE FOR Q1.3
        X_sums = X_binary.getnnz(axis=0)
        print("Number of reviews for Euclidean distance:")
        print(X_sums[:][103866])
        print(X_sums[:][103865])
        print(X_sums[:][98897])
        print(X_sums[:][72226])
        print(X_sums[:][102810])
        print("Number of reviews for cosine similarity:")
        print(X_sums[:][103866])
        print(X_sums[:][103867])
        print(X_sums[:][103865])
        print(X_sums[:][98068])
        print(X_sums[:][98066])

    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.png")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        dataPoints = np.ones(400)
        outliers = np.full((100,), 0.1)
        data = np.concatenate((dataPoints, outliers), axis=0)
        z = np.diag(data)

        model = linear_model.WeightedLeastSquares()
        model.fit(X,y,z)
        print(model.w)

        utils.test_and_plot(model, X, y, title="Weighted Least Squares", filename="least_squares_outliers_weighted.png")

    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.png")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.png")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        model = linear_model.LeastSquaresBias()
        model.fit(X, y)

        utils.test_and_plot(model, X, y, Xtest, ytest, title="Least Squares, bias",filename="least_squares_bias.png")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X, y)
            utils.test_and_plot(model, X, y, Xtest, ytest, title="Least Squares, poly",filename="least_squares_poly.png")

    else:
        print("Unknown question: %s" % question)

