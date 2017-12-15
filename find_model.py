import pandas as pd
import pygam
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
from sklearn import tree
from sklearn import linear_model
from sklearn import metrics
import math
import pydotplus 
from IPython.display import Image  
from sklearn import ensemble
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import statistics as st

def setup_models():
    models = []
    models.append(tree.DecisionTreeRegressor())
    original_params = {'n_estimators': 1000, 'max_leaf_nodes': 17, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}
    setting = {'learning_rate': 0.1, 'subsample': 1.0}
    params = dict(original_params)
    params.update(setting)
    gbr = ensemble.GradientBoostingRegressor(**params)
    models.append(gbr)
    svr = svm.SVR()
    models.append(svr)
    models.append(linear_model.LinearRegression())
    models.append(linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]))
    models.append(linear_model.LassoCV(alphas=[0.01, 0.1, 1.0, 10.0]))
    models.append(linear_model.MultiTaskLassoCV(alphas=[0.01, 0.1, 1.0, 10.0]))
    models.append(linear_model.ElasticNetCV(alphas=[0.01, 0.1, 1.0, 10.0]))
    models.append(linear_model.MultiTaskElasticNetCV(alphas=[0.01, 0.1, 1.0, 10.0]))
    models.append(linear_model.BayesianRidge())
    models.append(linear_model.SGDRegressor())
    models.append(linear_model.PassiveAggressiveRegressor())
    models.append(linear_model.RANSACRegressor())
    models.append(linear_model.TheilSenRegressor())
    models.append(linear_model.HuberRegressor())
    return models
    
def add_polynomial_features(models):
    poly_models = []
    for degree in range(2, 5):
        for model in models:
            new_model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                                 ('model', model)])
            poly_models.append(new_model)
    return poly_models
    
def find_model(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                        test_size=0.3, 
                                                                        random_state=42)
    rmses = []
    models = setup_models()
    poly_models = add_polynomial_features(models)
    models += poly_models
    for model in models:
        fitted_model = model.fit(X_train, y_train)
        predictions = fitted_model.predict(X_test)
        scores_applied = model.predict(X)
        rmses.append([predictions, math.sqrt(metrics.mean_squared_error(y_test.values, predictions))])
    return rmses

def choose_model(rmses):
    minimum = rmses[0][1]
    best_model_index = 0 
    for index, rmse in enumerate(rmses):
        if rmse[1] < minimum:
            minimum = rmse[1]
            best_model_index = index
    return rmses[best_model_index][0]

