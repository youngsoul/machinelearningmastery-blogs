# example of auto-sklearn for the insurance regression dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_absolute_error as auto_mean_absolute_error

if __name__ == '__main__':

    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
    dataframe = read_csv(url, header=None)
    # split into input and output elements
    data = dataframe.values
    data = data.astype('float32')
    X, y = data[:, :-1], data[:, -1]
    # split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    # define search
    model = AutoSklearnRegressor(time_left_for_this_task=20*60, per_run_time_limit=45, n_jobs=6, metric=auto_mean_absolute_error)
    # perform the search
    model.fit(X_train, y_train)
    # summarize
    print(model.sprint_statistics())
    # evaluate best model
    y_hat = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_hat)
    print("MAE: %.3f" % mae)

    print("Show models")
    models_def = model.show_models()
    print(models_def)