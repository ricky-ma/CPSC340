# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# our code
import utils
from knn import KNN
from naive_bayes import NaiveBayes
from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest
from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1, 15)  # depths to try

        my_tree_training_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_training_errors[i] = np.mean(y_pred != y)
        plt.plot(depths, my_tree_training_errors, label="trainingerrorrate")

        my_tree_testing_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X_test, y_test)
            y_pred = model.predict(X_test)
            my_tree_testing_errors[i] = np.mean(y_pred != y_test)
        plt.plot(depths, my_tree_testing_errors, label="testingerrorrate")

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_1_tree_errors.pdf")
        plt.savefig(fname)


    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape
        depths = np.arange(1, 15)  # depths to try

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        my_tree_training_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            my_tree_training_errors[i] = np.mean(y_pred != y_train)
        plt.plot(depths, my_tree_training_errors, label="trainingerrorrate")

        my_tree_testing_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X_test, y_test)
            y_pred = model.predict(X_test)
            my_tree_testing_errors[i] = np.mean(y_pred != y_test)
        plt.plot(depths, my_tree_testing_errors, label="testingerrorrate")

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_2_tree_errors.pdf")
        plt.savefig(fname)



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]

        print(wordlist[50])

        counter = 0
        temp_wordlist = []
        for word in X[500]:
            if word == 1:
                temp_wordlist.append(wordlist[counter])
                counter = counter+1
            else:
                counter = counter+1
        print(temp_wordlist)
        print(groupnames[y[500]])


    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        model2 = BernoulliNB()
        model2.fit(X, y)
        y_pred2 = model2.predict(X_valid)
        v_error2 = np.mean(y_pred2 != y_valid)
        print("Naive Bayes (sklearn) validation error: %.3f" % v_error2)


    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        model = KNN(k=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

        utils.plotClassifier(model, Xtest, ytest)
        fname = os.path.join("..", "figs", "q3.3_KNN.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        model = KNeighborsClassifier(1)
        model.fit(X, y)
        y_pred = model.predict(X)
        training_error = np.mean(y_pred != y)
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "q3.3_KNeighbors.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("Random tree")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("Random forest")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))
        print("sklearn random forest")
        evaluate_model(RandomForestClassifier(n_estimators=100))


    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        min_error = 9999999
        min_model = None
        for i in range(0, 50):
            model = Kmeans(4)
            model.fit(X)
            error = model.error(X)
            if error < min_error:
                min_model = model
                min_error = error
        print("Minimum error: " + str(min_error))
        plt.scatter(X[:,0], X[:,1], c=min_model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_good.png")
        plt.savefig(fname)
        print("Figure saved as ", fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']



    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        min_error = 123456123
        model_min = None
        track_error = np.zeros(10)
        for j in range(1, 11):
            for i in range(0, 50):
                model = Kmeans(k=j)
                model.fit(X)
                error_val = model.error(X)
                if error_val < min_error:
                    model_min = model
                    min_error = error_val

            track_error[j - 1] = min_error
            min_error = 123456123

        k_val = np.arange(1, 11)

        plt.plot(k_val, track_error, label="Minimum error", linestyle='-', marker='o')
        plt.xlabel("k")
        plt.ylabel("Minimum Error")
        plt.xticks(np.arange(min(k_val), max(k_val) + 1, 1.0))
        plt.legend()
        fname = "../figs/KValueMinimumError.png"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        plt.show()



    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=15, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
