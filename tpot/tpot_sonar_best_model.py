import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':

    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv'

    # tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
    tpot_data = pd.read_csv(url)
    data = tpot_data.values

    X, y = data[:, :-1], data[:, -1]
    y = LabelEncoder().fit_transform(y.astype('str'))

    # features = tpot_data.drop('target', axis=1)
    training_features, testing_features, training_target, testing_target = \
                train_test_split(X, y, random_state=1)

    # Average CV score on the training set was: 0.8764285714285714
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=3, min_samples_split=2, n_estimators=100)),
        GradientBoostingClassifier(learning_rate=1.0, max_depth=4, max_features=0.2, min_samples_leaf=9, min_samples_split=6, n_estimators=100, subsample=1.0)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 1)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    # make a prediction on a new row of data
    row = [0.0200, 0.0371, 0.0428, 0.0207, 0.0954, 0.0986, 0.1539, 0.1601, 0.3109, 0.2111, 0.1609, 0.1582, 0.2238,
           0.0645, 0.0660, 0.2273, 0.3100, 0.2999, 0.5078, 0.4797, 0.5783, 0.5071, 0.4328, 0.5550, 0.6711, 0.6415,
           0.7104, 0.8080, 0.6791, 0.3857, 0.1307, 0.2604, 0.5121, 0.7547, 0.8537, 0.8507, 0.6692, 0.6097, 0.4943,
           0.2744, 0.0510, 0.2834, 0.2825, 0.4256, 0.2641, 0.1386, 0.1051, 0.1343, 0.0383, 0.0324, 0.0232, 0.0027,
           0.0065, 0.0159, 0.0072, 0.0167, 0.0180, 0.0084, 0.0090, 0.0032]
    yhat = exported_pipeline.predict([row])
    print('Predicted: %.3f' % yhat[0])
