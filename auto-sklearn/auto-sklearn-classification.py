# example of auto-sklearn for the sonar classification dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from autosklearn.classification import AutoSklearnClassifier

if __name__ == '__main__':

    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
    dataframe = read_csv(url, header=None)
    # print(dataframe.head())
    # split into input and output elements
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1]

    # minimally prepare dataset
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))

    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # define search
    model = AutoSklearnClassifier(time_left_for_this_task=10*60, per_run_time_limit=45, n_jobs=6)

    # perform the search
    model.fit(X_train, y_train)
    # summarize
    print(model.sprint_statistics())

    # get model and weights
    model_weights = model.get_models_with_weights()
    for model_weight in model_weights:
        print(model_weight)

    print("Show models")
    models_def = model.show_models()
    print(models_def)

    # evaluate best model
    y_hat = model.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    print("Test Dataset Accuracy: %.3f" % acc)