#!/usr/bin/env python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

def loadData():
    """
    Function: load_data()
    ---------------------

    Loads train data
    data: train data

    returns data(loaded file var)
    """
    data = pd.read_csv("train.csv")
    return data

def cleanData(data):
    """
    Function: cleanData
    --------------------

    Takes the loaded file, splits into categorical and numerical data.
    Then fills in missing values, encodes categorical data and combines
    the two dataframes back into one.

    cat: array of object columns
    num: array of numerical columns

    dataCat: var for cat[]
    dataNum: var for num[]

    le: label encoder
    y: transformed values of dataCat categorical dataframe

    result: return value of concatenated dataFrames

    returns result
    """
    cat = []
    num = []
    for col in data.columns.values:
        if data[col].dtype == "object":
            cat.append(col)
        else:
            num.append(col)

    dataCat = data[cat]
    dataNum = data[num]

    dataNum = dataNum.fillna(dataNum.median())
    dataCat = dataCat.fillna("none")

    for col in dataCat.columns.values:
        le = LabelEncoder()
        y = le.fit_transform(dataCat[col])
        dataCat[col] = y

    dataNum = pd.DataFrame(dataNum)
    dataCat = pd.DataFrame(dataCat)
    result = pd.concat([dataCat, dataNum], axis=1)

    return result

def buildModel(result):
    """
    Function: buildModel()
    --------------------

    Takes the resulting dataFrame, creates x and y values to train the model,
    creates the model, scores and fits it

    y: sale price of the houses as a matrix
    x: the rest of the data with dropped sale price

    model: Random Forest Regressor model

    score: the cross_val_score of the model

    returns model
    """
    y = pd.DataFrame(result, columns=['SalePrice']).as_matrix().ravel()
    x = result.drop(['Id','SalePrice'], 1).as_matrix()

    model = RandomForestRegressor(100)
    score = np.mean(cross_val_score(model, x, y, cv=10))
    print(score)
    model.fit(x, y)

    return model


def loadTestData():
    """
    Function: loadTestData()
    ------------------------

    Loads the test data and returns the dataframe

    test_data: loaded test data

    return test_data(loaded test data)
    """
    test_data = pd.read_csv("test.csv")

    return test_data


def predictAndPrint(test_data, model):
    """
    Function: predictAndPrint()
    -----------------------

    Runs the test data through the model to produce prediction and creates submission
    file

    test_y: id column of the dataframe
    test_x: the rest of the data to predict

    final: the final prediction of the model

    sub: compiled information for submission

    returns nothing
    """

    test_y = pd.DataFrame(test_data, columns=['Id']).as_matrix().ravel()
    test_x = test_data.drop('Id', 1).as_matrix()

    final = model.predict(test_x)
    sub = pd.DataFrame(final, index=test_y, columns=["SalePrice"])
    sub.index.names = ['Id']
    sub.to_csv('submission2.csv')


def main():
    """
    Function: main()

    Main function of the program to execute the code

    data: train data
    result: cleaned train data
    test_data: test data

    returns nothing
    """

    data = loadData()
    result = cleanData(data)
    model = buildModel(result)
    test_data = loadTestData()
    predictAndPrint(test_data, model)


if __name__ == "__main__":
    main()
