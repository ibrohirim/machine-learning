import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical


def main():

    """
    Function: main
    --------------

    The digit recognizer program

    train_data: data imported from a file for training

    y: labels of the data to use for testing the trained model

    x: core data consiting of RGB vallues for wach pixel of the image

    model: neural network model to be trained

    loss_and_metrics: evaulation of the model

    test_data: test data loaded from a file to be predicted

    predict: prediction results of the test data run through the NN

    list_val: reshaping of the first column of predicted data

    array_val: list_val turned to an array

    col1: reshaped final value of array_val and firdt column of output file

    col2: predicted values and second column of outpur file

    df: dataframe object of output file

    returns: nothing

    """

    # Read in the data
    train_data = pd.read_csv('train.csv')

    # Split the training and test data inputs and outputs
    y = pd.DataFrame(train_data, columns=['label']).as_matrix()
    x = train_data.drop('label', 1).as_matrix()

    # Convert output to categorical data
    y = to_categorical(y, 10)

    # Split the data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    # Create the model
    model = Sequential()

    # Build the layers
    model.add(Dense(output_dim=64, input_dim=784))
    model.add(Activation('relu'))
    model.add(Dropout(0.15))

    for i in range(0, 4):
        model.add(Dense(output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dropout(0.33))

    model.add(Dense(output_dim=10))
    model.add(Activation('softmax'))
    model.add(Dropout(0.15))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer="adadelta",
                  metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, nb_epoch=25)

    # Evaulate the model
    loss_and_metrics = model.evaluate(x_test, y_test)
    print(loss_and_metrics)

    # Import test data
    test_data = pd.read_csv('test.csv').as_matrix()
    predict = model.predict(test_data)

    # Format data and save to file
    list_val = range(1, predict.shape[0]+1)
    array_val = np.asarray(list_val)
    col1 = array_val.reshape(len(list_val), 1)

    col2 = np.argmax(predict, axis=1)
    col2 = col2.reshape(col2.shape[0], 1)

    array = np.concatenate((col1, col2), axis=1)
    col_headers = ['ImageId', 'Label']

    df = pd.DataFrame(array, columns=col_headers)
    df.set_index('ImageId', inplace=True)
    df.to_csv('submission.csv')

if __name__ == '__main__':
    main()
