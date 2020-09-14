# example of tpot for the sonar classification dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from tpot import TPOTClassifier

if __name__ == '__main__':

    # load dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'
    dataframe = read_csv(url, header=None)
    # split into input and output elements
    data = dataframe.values
    X, y = data[:, :-1], data[:, -1]
    # minimally prepare dataset
    X = X.astype('float32')
    y = LabelEncoder().fit_transform(y.astype('str'))
    # define model evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search
    model = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
    # perform the search
    model.fit(X, y)
    # export the best model
    model.export('tpot_sonar_best_model.py')