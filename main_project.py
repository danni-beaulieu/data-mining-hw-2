import math

import pandas as pd
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, CategoricalNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


def doKFold(X, y, models, names, k):
    num_models = len(models)

    accuracy_array_tr = [[None for i in range(k)] for j in range(num_models)]
    accuracy_array_te = [[None for i in range(k)] for j in range(num_models)]
    average_accuracy_tr = [None] * num_models
    average_accuracy_te = [None] * num_models

    kfold = KFold(n_splits=k, shuffle=True, random_state=111)
    folds = [next(kfold.split(X)) for i in range(k)]

    for split_i in range(k):
        X_tr = X[folds[split_i][0]]
        X_te = X[folds[split_i][1]]
        y_tr = y[folds[split_i][0]]
        y_te = y[folds[split_i][1]]

        for j in range(len(models)):
            models[j].fit(X_tr, y_tr)
            accuracy_array_tr[j][split_i] = models[j].score(X_tr, y_tr)
            accuracy_array_te[j][split_i] = models[j].score(X_te, y_te)

    bestparam = 0
    avgaccuracy = -1 * math.inf
    for j in range(num_models):
        average_accuracy_tr[j] = sum(accuracy_array_tr[j]) / k
        average_accuracy_te[j] = sum(accuracy_array_te[j]) / k
        print('Model : ', names[j])
        print('Accuracy of each fold - {}'.format(accuracy_array_te[j]))
        print('Avg accuracy error : {}'.format(average_accuracy_te[j]))
        if (average_accuracy_te[j] > avgaccuracy):
            bestparam = j
            avgaccuracy = average_accuracy_te[j]

    bestmodel = models[bestparam].fit(X, y)

    return bestmodel, average_accuracy_te


def do_split(df, frTest):
    test = df.sample(frac=frTest, axis=0, replace=False)
    train = df.drop(index=test.index)
    return train, test


print("BEGIN")


# Loading and shaping

df = pd.read_csv("letter-recognition.data", header=None)
print(df.shape)

hk_df = df.loc[df[0].isin(['H','K'])]
print(hk_df.shape)

ym_df = df.loc[df[0].isin(['Y','M'])]
print(ym_df.shape)

od_df = df.loc[df[0].isin(['O','D'])]
print(od_df.shape)

hk_train_df, hk_test_df = do_split(hk_df, 0.1)
ym_train_df, ym_test_df = do_split(ym_df, 0.1)
od_train_df, od_test_df = do_split(od_df, 0.1)

hk_raw_data_train = hk_train_df.to_numpy()
hk_raw_data_test = hk_test_df.to_numpy()
hk_X_train = hk_raw_data_train[:, 1:(hk_raw_data_train.shape[1])]
hk_y_train = hk_raw_data_train[:, 0]
hk_X_test = hk_raw_data_test[:, 1:(hk_raw_data_test.shape[1])]
hk_y_test = hk_raw_data_test[:, 0]

ym_raw_data_train = ym_train_df.to_numpy()
ym_raw_data_test = ym_test_df.to_numpy()
ym_X_train = ym_raw_data_train[:, 1:(ym_raw_data_train.shape[1])]
ym_y_train = ym_raw_data_train[:, 0]
ym_X_test = ym_raw_data_test[:, 1:(ym_raw_data_test.shape[1])]
ym_y_test = ym_raw_data_test[:, 0]

od_raw_data_train = od_train_df.to_numpy()
od_raw_data_test = od_test_df.to_numpy()
od_X_train = od_raw_data_train[:, 1:(od_raw_data_train.shape[1])]
od_y_train = od_raw_data_train[:, 0]
od_X_test = od_raw_data_test[:, 1:(od_raw_data_test.shape[1])]
od_y_test = od_raw_data_test[:, 0]


# Preprocessing pipeline

hk_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(chi2, k=4))]).fit(hk_X_train, hk_y_train)
hk_X_train_pp = hk_pipeline.transform(hk_X_train)
hk_X_test_pp = hk_pipeline.transform(hk_X_test)

ym_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(chi2, k=4))]).fit(ym_X_train, ym_y_train)
ym_X_train_pp = ym_pipeline.transform(ym_X_train)
ym_X_test_pp = ym_pipeline.transform(ym_X_test)

od_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(chi2, k=4))]).fit(od_X_train, od_y_train)
od_X_train_pp = od_pipeline.transform(od_X_train)
od_X_test_pp = od_pipeline.transform(od_X_test)


# 5-fold cross-validation (non-processed)

hk_bestNB, hkNB_accuracies = doKFold(hk_X_train, hk_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)
ym_bestNB, ymNB_accuracies = doKFold(ym_X_train, ym_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)
od_bestNB, odNB_accuracies = doKFold(od_X_train, od_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)

hk_bestSVM, hkSVM_accuracies = doKFold(hk_X_train, hk_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)
ym_bestSVM, ymSVM_accuracies = doKFold(ym_X_train, ym_y_train,
                                   [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')],
                                   ["Linear SVM", "Polynomial SVM", "RBF SVM"], 5)
od_bestSVM, odSVM_accuracies = doKFold(od_X_train, od_y_train,
                                   [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')],
                                   ["Linear SVM", "Polynomial SVM", "RBF SVM"], 5)


# 5-fold cross-validation (processed)

hk_bestNB_pp, hkNB_accuracies_pp = doKFold(hk_X_train_pp, hk_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)
ym_bestNB_pp, ymNB_accuracies_pp = doKFold(ym_X_train_pp, ym_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)
od_bestNB_pp, odNB_accuracies_pp = doKFold(od_X_train_pp, od_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)

hk_bestSVM_pp, hkSVM_accuracies_pp = doKFold(hk_X_train_pp, hk_y_train,
                                   [MultinomialNB(), BernoulliNB(), GaussianNB()],
                                   ["Multinomial NB", "Bernoulli NB", "Gaussian NB"], 5)
ym_bestSVM_pp, ymSVM_accuracies_pp = doKFold(ym_X_train_pp, ym_y_train,
                                   [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')],
                                   ["Linear SVM", "Polynomial SVM", "RBF SVM"], 5)
od_bestSVM_pp, odSVM_accuracies_pp = doKFold(od_X_train_pp, od_y_train,
                                   [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')],
                                   ["Linear SVM", "Polynomial SVM", "RBF SVM"], 5)

# Score test

print(hk_bestNB.score(hk_X_test, hk_y_test))
print(ym_bestNB.score(ym_X_test, ym_y_test))
print(od_bestNB.score(od_X_test, od_y_test))

print(hk_bestSVM.score(hk_X_test, hk_y_test))
print(ym_bestSVM.score(ym_X_test, ym_y_test))
print(od_bestSVM.score(od_X_test, od_y_test))

print(hk_bestNB_pp.score(hk_X_test_pp, hk_y_test))
print(ym_bestNB_pp.score(ym_X_test_pp, ym_y_test))
print(od_bestNB_pp.score(od_X_test_pp, od_y_test))

print(hk_bestSVM_pp.score(hk_X_test_pp, hk_y_test))
print(ym_bestSVM_pp.score(ym_X_test_pp, ym_y_test))
print(od_bestSVM_pp.score(od_X_test_pp, od_y_test))

print("END")
