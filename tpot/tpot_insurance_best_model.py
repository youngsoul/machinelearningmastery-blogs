import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

if __name__ == '__main__':

    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
    dataframe = pd.read_csv(url, header=None)
    # split into input and output elements
    data = dataframe.values
    data = data.astype('float32')
    X, y = data[:, :-1], data[:, -1]

    training_features, testing_features, training_target, testing_target = \
                train_test_split(X, y, random_state=1)

    # Average CV score on the training set was: -29.116294532472594
    exported_pipeline = LinearSVR(C=15.0, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.001)
    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 1)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    # make a prediction on a new row of data
    row = [108]
    yhat = exported_pipeline.predict([row])
    print('Predicted: %.3f' % yhat[0])