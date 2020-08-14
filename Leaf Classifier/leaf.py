#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def load_data():
    """
    Function: load_data
    -------------------

    Loads train data

    leaf: the loaded file

    returns the loaded file
    """
    leaf = pd.read_csv('train.csv')
    return leaf


def define_y(leaf):
    """
    Function: define_y
    ------------------

    Takes in the loaded file and defines the target data y

    y: target data

    return y
    """
    Y = pd.DataFrame(leaf, columns=['species']).as_matrix().ravel()
    return Y


def define_x(leaf):
    """
    Function: define_x
    ------------------

    Takes in the loaded file and defines the data to be used as training x

    x: training data

    returns x
    """
    X = leaf.drop(['id', 'species'], 1).as_matrix()
    return X


def transform(y):
    """
    Function: transform
    -------------------

    Transforms the data

    le: loaded LabelEncoder object
    y: the transformed target area

    returns y
    """
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    return y, le


def build_model(x, y):
    """
    Function: build_model
    ---------------------

    Builds the model and trains it

    model: the RandomForest model that we'll train
    score: the cross validation score of the k-folds

    returns model
    """
    model = RandomForestClassifier(60)
    score = np.mean(cross_val_score(model, x, y, cv=10))
    model.fit(x, y)

    return model, score


def load_predict_save(model, le):
    """
    Function: load_predict
    ------------------------

    Loads the test data and makes a prediction and saves the output file to csv

    test_leaf: the test data to be run through the file
    test_ids: dataframe of the loaded test data as a matrix
    predict: the prediction of the test data of the model
    sub: dataframe to be saved to csv
    """
    test_leaf = pd.read_csv('test.csv')
    test_ids = pd.DataFrame(test_leaf, columns=['id']).as_matrix().ravel()
    test_leaf = test_leaf.drop('id', 1).as_matrix()
    predict = model.predict_proba(test_leaf)
    sub = pd.DataFrame(predict, index=test_ids, columns=le.classes_)
    sub.index.names = ['id']
    sub.to_csv('sybmission_leaf.csv')


def main():
    """
    Function: main
    --------------
    main function to execute the code

    leaf: the loaded data file
    y: target of date file
    x: date of datafile
    le: classes of datafile
    model: the model to predict
    score: score of the predicted model
    """
    leaf = load_data()
    y = define_y(leaf)
    x = define_x(leaf)
    y, le = transform(y)

    model, score = build_model(x, y)
    print(score)
    load_predict_save(model, le)


if __name__ == "__main__":
    main()
