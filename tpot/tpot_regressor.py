# example of tpot for the insurance regression dataset
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor

if __name__ == '__main__':

    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-insurance.csv'
    dataframe = read_csv(url, header=None)
    # split into input and output elements
    data = dataframe.values
    data = data.astype('float32')
    X, y = data[:, :-1], data[:, -1]
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search
    model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
    # perform the search
    model.fit(X, y)
    # export the best model
    model.export('tpot_insurance_best_model.py')