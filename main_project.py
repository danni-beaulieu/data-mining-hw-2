import math
import os
import sys
from time import process_time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.model_selection import KFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB, CategoricalNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


def doKFold(X, y, models, names, k):
    num_models = len(models)

    accuracy_array_tr = [[None for i in range(k)] for j in range(num_models)]
    accuracy_array_te = [[None for i in range(k)] for j in range(num_models)]
    average_accuracy_tr = [None] * num_models
    average_accuracy_te = [None] * num_models

    kfold = KFold(n_splits=k, shuffle=True)
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

    start = process_time()
    bestmodel = models[bestparam].fit(X, y)
    end = process_time()

    return bestmodel, accuracy_array_te, (end - start), names[bestparam]


def do_split(df, frTest):
    test = df.sample(frac=frTest, axis=0, replace=False, random_state=111)
    train = df.drop(index=test.index)
    return train, test


original_stdout = sys.stdout
os.makedirs('./non-preprocessed', exist_ok=True)
os.makedirs('./preprocessed', exist_ok=True)
with open('output.txt', 'w+') as np_out:
    sys.stdout = np_out
    print("BEGIN")

    # Loading and shaping

    df = pd.read_csv("letter-recognition.data", header=None)

    hk_df = df.loc[df[0].isin(['H', 'K'])]
    ym_df = df.loc[df[0].isin(['Y', 'M'])]
    od_df = df.loc[df[0].isin(['O', 'D'])]

    hk_train_df, hk_test_df = do_split(hk_df, 0.1)
    ym_train_df, ym_test_df = do_split(ym_df, 0.1)
    od_train_df, od_test_df = do_split(od_df, 0.1)
    all_train_df, all_test_df = do_split(df, 0.1)

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

    all_raw_data_train = all_train_df.to_numpy()
    all_raw_data_test = all_test_df.to_numpy()
    all_X_train = all_raw_data_train[:, 1:(all_raw_data_train.shape[1])]
    all_y_train = all_raw_data_train[:, 0]
    all_X_test = all_raw_data_test[:, 1:(all_raw_data_test.shape[1])]
    all_y_test = all_raw_data_test[:, 0]

    # Preprocessing pipeline

    hk_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(mutual_info_classif, k=4))]).fit(
        hk_X_train, hk_y_train)
    hk_X_train_pp = hk_pipeline.transform(hk_X_train)
    hk_X_test_pp = hk_pipeline.transform(hk_X_test)

    ym_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(mutual_info_classif, k=4))]).fit(
        ym_X_train, ym_y_train)
    ym_X_train_pp = ym_pipeline.transform(ym_X_train)
    ym_X_test_pp = ym_pipeline.transform(ym_X_test)

    od_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(mutual_info_classif, k=4))]).fit(
        od_X_train, od_y_train)
    od_X_train_pp = od_pipeline.transform(od_X_train)
    od_X_test_pp = od_pipeline.transform(od_X_test)

    all_pipeline = Pipeline([('scaling', MinMaxScaler()), ('reduction', SelectKBest(mutual_info_classif, k=4))]).fit(
        all_X_train, all_y_train)
    all_X_train_pp = all_pipeline.transform(all_X_train)
    all_X_test_pp = all_pipeline.transform(all_X_test)

    # 5-fold cross-validation (non-processed)

    namesNB = ["Multinomial NB", "Bernoulli NB", "Gaussian NB"]
    namesSVM = ["Linear SVM", "Polynomial SVM", "RBF SVM"]
    namesKNN = ["1-NN", "2-NN", "3-NN", "4-NN", "5-NN"]

    hk_bestNB, hkNB_accuracies, hkNB_time, hkNB_name = doKFold(hk_X_train, hk_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)
    ym_bestNB, ymNB_accuracies, ymNB_time, ymNB_name = doKFold(ym_X_train, ym_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)
    od_bestNB, odNB_accuracies, odNB_time, odNB_name = doKFold(od_X_train, od_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)
    all_bestNB, allNB_accuracies, allNB_time, allNB_name = doKFold(all_X_train, all_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)

    hk_bestSVM, hkSVM_accuracies, hkSVM_time, hkSVM_name = doKFold(hk_X_train, hk_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)
    ym_bestSVM, ymSVM_accuracies, ymSVM_time, ymSVM_name = doKFold(ym_X_train, ym_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)
    od_bestSVM, odSVM_accuracies, odSVM_time, odSVM_name = doKFold(od_X_train, od_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)
    all_bestSVM, allSVM_accuracies, allSVM_time, allSVM_name = doKFold(all_X_train, all_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)

    hk_bestKNN, hkKNN_accuracies, hkKNN_time, hkKNN_name = doKFold(hk_X_train, hk_y_train,
                                                                   [KNeighborsClassifier(n_neighbors=1),
                                                                    KNeighborsClassifier(n_neighbors=2),
                                                                    KNeighborsClassifier(n_neighbors=3),
                                                                    KNeighborsClassifier(n_neighbors=4),
                                                                    KNeighborsClassifier(n_neighbors=5)], namesKNN, 5)
    ym_bestKNN, ymKNN_accuracies, ymKNN_time, ymKNN_name = doKFold(ym_X_train, ym_y_train,
                                                               [KNeighborsClassifier(n_neighbors=1),
                                                                KNeighborsClassifier(n_neighbors=2),
                                                                KNeighborsClassifier(n_neighbors=3),
                                                                KNeighborsClassifier(n_neighbors=4),
                                                                KNeighborsClassifier(n_neighbors=5)], namesKNN, 5)
    od_bestKNN, odKNN_accuracies, odKNN_time, odKNN_name = doKFold(od_X_train, od_y_train,
                                                               [KNeighborsClassifier(n_neighbors=1),
                                                                KNeighborsClassifier(n_neighbors=2),
                                                                KNeighborsClassifier(n_neighbors=3),
                                                                KNeighborsClassifier(n_neighbors=4),
                                                                KNeighborsClassifier(n_neighbors=5)], namesKNN, 5)
    all_bestKNN, allKNN_accuracies, allKNN_time, allKNN_name = doKFold(all_X_train, all_y_train,
                                                                   [KNeighborsClassifier(n_neighbors=1),
                                                                    KNeighborsClassifier(n_neighbors=2),
                                                                    KNeighborsClassifier(n_neighbors=3),
                                                                    KNeighborsClassifier(n_neighbors=4),
                                                                    KNeighborsClassifier(n_neighbors=5)], namesKNN, 5)

    # 5-fold cross-validation (processed)

    hk_bestNB_pp, hkNB_accuracies_pp, hkNB_time_pp, hkNB_name_pp = doKFold(hk_X_train_pp, hk_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)
    ym_bestNB_pp, ymNB_accuracies_pp, ymNB_time_pp, ymNB_name_pp = doKFold(ym_X_train_pp, ym_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)
    od_bestNB_pp, odNB_accuracies_pp, odNB_time_pp, odNB_name_pp = doKFold(od_X_train_pp, od_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)
    all_bestNB_pp, allNB_accuracies_pp, allNB_time_pp, allNB_name_pp = doKFold(all_X_train_pp, all_y_train,
                                                            [MultinomialNB(), BernoulliNB(), GaussianNB()], namesNB, 5)

    hk_bestSVM_pp, hkSVM_accuracies_pp, hkSVM_time_pp, hkSVM_name_pp = doKFold(hk_X_train_pp, hk_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)
    ym_bestSVM_pp, ymSVM_accuracies_pp, ymSVM_time_pp, ymSVM_name_pp = doKFold(ym_X_train_pp, ym_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)
    od_bestSVM_pp, odSVM_accuracies_pp, odSVM_time_pp, odSVM_name_pp = doKFold(od_X_train_pp, od_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)
    all_bestSVM_pp, allSVM_accuracies_pp, allSVM_time_pp, allSVM_name_pp = doKFold(all_X_train_pp, all_y_train,
                                            [SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf')], namesSVM, 5)

    hk_bestKNN_pp, hkKNN_accuracies_pp, hkKNN_time_pp, hkKNN_name_pp = doKFold(hk_X_train_pp, hk_y_train,
                                                                               [KNeighborsClassifier(n_neighbors=1),
                                                                                KNeighborsClassifier(n_neighbors=2),
                                                                                KNeighborsClassifier(n_neighbors=3),
                                                                                KNeighborsClassifier(n_neighbors=4),
                                                                                KNeighborsClassifier(n_neighbors=5)],
                                                                               namesKNN, 5)
    ym_bestKNN_pp, ymKNN_accuracies_pp, ymKNN_time_pp, ymKNN_name_pp = doKFold(ym_X_train_pp, ym_y_train,
                                                                               [KNeighborsClassifier(n_neighbors=1),
                                                                                KNeighborsClassifier(n_neighbors=2),
                                                                                KNeighborsClassifier(n_neighbors=3),
                                                                                KNeighborsClassifier(n_neighbors=4),
                                                                                KNeighborsClassifier(n_neighbors=5)],
                                                                               namesKNN, 5)
    od_bestKNN_pp, odKNN_accuracies_pp, odKNN_time_pp, odKNN_name_pp = doKFold(od_X_train_pp, od_y_train,
                                                                               [KNeighborsClassifier(n_neighbors=1),
                                                                                KNeighborsClassifier(n_neighbors=2),
                                                                                KNeighborsClassifier(n_neighbors=3),
                                                                                KNeighborsClassifier(n_neighbors=4),
                                                                                KNeighborsClassifier(n_neighbors=5)],
                                                                               namesKNN, 5)
    all_bestKNN_pp, allKNN_accuracies_pp, allKNN_time_pp, allKNN_name_pp = doKFold(all_X_train_pp, all_y_train,
                                                                                [KNeighborsClassifier(n_neighbors=1),
                                                                                    KNeighborsClassifier(n_neighbors=2),
                                                                                    KNeighborsClassifier(n_neighbors=3),
                                                                                    KNeighborsClassifier(n_neighbors=4),
                                                                                    KNeighborsClassifier(
                                                                                        n_neighbors=5)], namesKNN, 5)

    # Plot and score test

    sns.set()

    sns.boxplot(data=pd.DataFrame(hkNB_accuracies, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="H/K NB Non-Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "H-K NB Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(ymNB_accuracies, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="Y/M NB Non-Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "Y-M NB Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(odNB_accuracies, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="O/D NB Non-Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "O-D NB Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(allNB_accuracies, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="All NB Non-Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "All NB Non-Processed" + '.png', bbox_inches='tight')
    plt.show()

    sns.boxplot(data=pd.DataFrame(hkSVM_accuracies, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="H/K SVM Non-Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "H-K SVM Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(ymSVM_accuracies, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="Y/M SVM Non-Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "Y-M SVM Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(odSVM_accuracies, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="O/D SVM Non-Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "O-D SVM Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(allSVM_accuracies, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="All SVM Non-Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "All SVM Non-Processed" + '.png', bbox_inches='tight')
    plt.show()

    sns.boxplot(data=pd.DataFrame(hkKNN_accuracies, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="H/K KNN Non-Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "H-K KNN Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(ymKNN_accuracies, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="Y/M KNN Non-Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "Y-M KNN Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(odKNN_accuracies, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="O/D KNN Non-Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "O-D KNN Non-Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(allKNN_accuracies, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="All KNN Non-Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./non-preprocessed/' + "All KNN Non-Processed" + '.png', bbox_inches='tight')
    plt.show()

    sns.boxplot(data=pd.DataFrame(hkNB_accuracies_pp, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="H/K NB Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "H-K NB Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(ymNB_accuracies_pp, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="Y/M NB Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "Y-M NB Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(odNB_accuracies_pp, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="O/D NB Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "O-D NB Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(allNB_accuracies_pp, namesNB).T, medianprops=dict(color="red", alpha=0.7)).set(title="All NB Processed")
    plt.xlabel("Distribution")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "All NB Processed" + '.png', bbox_inches='tight')
    plt.show()

    sns.boxplot(data=pd.DataFrame(hkSVM_accuracies_pp, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="H/K SVM Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "H-K SVM Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(ymSVM_accuracies_pp, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="Y/M SVM Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "Y-M SVM Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(odSVM_accuracies_pp, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="O/D SVM Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "O-D SVM Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(allSVM_accuracies_pp, namesSVM).T, medianprops=dict(color="red", alpha=0.7)).set(title="All SVM Processed")
    plt.xlabel("Kernel")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "All SVM Processed" + '.png', bbox_inches='tight')
    plt.show()

    sns.boxplot(data=pd.DataFrame(hkKNN_accuracies_pp, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="H/K KNN Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "H-K KNN Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(ymKNN_accuracies_pp, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="Y/M KNN Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "Y-M KNN Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(odKNN_accuracies_pp, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="O/D KNN Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "O-D KNN Processed" + '.png', bbox_inches='tight')
    plt.show()
    sns.boxplot(data=pd.DataFrame(allKNN_accuracies_pp, namesKNN).T, medianprops=dict(color="red", alpha=0.7)).set(title="All KNN Processed")
    plt.xlabel("Neighbors")
    plt.ylabel("Accuracy")
    plt.savefig('./preprocessed/' + "All KNN Processed" + '.png', bbox_inches='tight')
    plt.show()

    print("Model: {} - Final accuracy: {} - Run time: {}".format(hkNB_name, hk_bestNB.score(hk_X_test, hk_y_test),
                                                                 hkNB_time))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(ymNB_name, ym_bestNB.score(ym_X_test, ym_y_test),
                                                                 ymNB_time))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(odNB_name, od_bestNB.score(od_X_test, od_y_test),
                                                                 odNB_time))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(allNB_name, all_bestNB.score(all_X_test, all_y_test),
                                                                 allNB_time))

    print("Model: {} - Final accuracy: {} - Run time: {}".format(hkSVM_name, hk_bestSVM.score(hk_X_test, hk_y_test),
                                                                 hkSVM_time))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(ymSVM_name, ym_bestSVM.score(ym_X_test, ym_y_test),
                                                                 ymSVM_time))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(odSVM_name, od_bestSVM.score(od_X_test, od_y_test),
                                                                 odSVM_time))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(allSVM_name, all_bestSVM.score(all_X_test, all_y_test),
                                                                 allSVM_time))

    print("Model: {} - Final accuracy: {} - Run time: {}".format(hkNB_name_pp,
                                                                 hk_bestNB_pp.score(hk_X_test_pp, hk_y_test),
                                                                 hkNB_time_pp))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(ymNB_name_pp,
                                                                 ym_bestNB_pp.score(ym_X_test_pp, ym_y_test),
                                                                 ymNB_time_pp))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(odNB_name_pp,
                                                                 od_bestNB_pp.score(od_X_test_pp, od_y_test),
                                                                 odNB_time_pp))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(allNB_name_pp,
                                                                 all_bestNB_pp.score(all_X_test_pp, all_y_test),
                                                                 allNB_time_pp))

    print("Model: {} - Final accuracy: {} - Run time: {}".format(hkSVM_name_pp,
                                                                 hk_bestSVM_pp.score(hk_X_test_pp, hk_y_test),
                                                                 hkSVM_time_pp))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(ymSVM_name_pp,
                                                                 ym_bestSVM_pp.score(ym_X_test_pp, ym_y_test),
                                                                 ymSVM_time_pp))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(odSVM_name_pp,
                                                                 od_bestSVM_pp.score(od_X_test_pp, od_y_test),
                                                                 odSVM_time_pp))
    print("Model: {} - Final accuracy: {} - Run time: {}".format(allSVM_name_pp,
                                                                 all_bestSVM_pp.score(all_X_test_pp, all_y_test),
                                                                 allSVM_time_pp))

    print("END")
    sys.stdout = original_stdout
    np_out.close()
