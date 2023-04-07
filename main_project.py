import pandas as pd
from sklearn.naive_bayes import BernoulliNB, GaussianNB, CategoricalNB, MultinomialNB
from sklearn.svm import SVC


def do_split(df, frTest):
    test = df.sample(frac=frTest, axis=0, replace=False)
    train = df.drop(index=test.index)
    return train, test


print("BEGIN")

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

hk_mNB = MultinomialNB()
hk_mNB.fit(hk_X_train, hk_y_train)
hk_mNB_p = hk_mNB.predict(hk_X_train)
print(hk_mNB.score(hk_X_train, hk_y_train))

hk_cNB = CategoricalNB(min_categories=2)
hk_cNB.fit(hk_X_train, hk_y_train)
hk_cNB_p = hk_cNB.predict(hk_X_train)
print(hk_cNB.score(hk_X_train, hk_y_train))

hk_gNB = GaussianNB()
hk_gNB.fit(hk_X_train, hk_y_train)
hk_gNB_p = hk_gNB.predict(hk_X_train)
print(hk_gNB.score(hk_X_train, hk_y_train))

hk_lSVM = SVC(kernel='linear')
hk_lSVM.fit(hk_X_train, hk_y_train)
hk_lSVM_p = hk_lSVM.predict(hk_X_train)
print(hk_lSVM.score(hk_X_train, hk_y_train))

hk_pSVM = SVC(kernel='poly')
hk_pSVM.fit(hk_X_train, hk_y_train)
hk_pSVM_p = hk_pSVM.predict(hk_X_train)
print(hk_pSVM.score(hk_X_train, hk_y_train))

hk_rSVM = SVC(kernel='rbf')
hk_rSVM.fit(hk_X_train, hk_y_train)
hk_rSVM_p = hk_rSVM.predict(hk_X_train)
print(hk_rSVM.score(hk_X_train, hk_y_train))

ym_mNB = MultinomialNB()
ym_mNB.fit(ym_X_train, ym_y_train)
ym_mNB_p = ym_mNB.predict(ym_X_train)
print(ym_mNB.score(ym_X_train, ym_y_train))

ym_cNB = CategoricalNB(min_categories=2)
ym_cNB.fit(ym_X_train, ym_y_train)
ym_cNB_p = ym_cNB.predict(ym_X_train)
print(ym_cNB.score(ym_X_train, ym_y_train))

ym_gNB = GaussianNB()
ym_gNB.fit(ym_X_train, ym_y_train)
ym_gNB_p = ym_gNB.predict(ym_X_train)
print(ym_gNB.score(ym_X_train, ym_y_train))

ym_lSVM = SVC(kernel='linear')
ym_lSVM.fit(ym_X_train, ym_y_train)
ym_lSVM_p = ym_lSVM.predict(ym_X_train)
print(ym_lSVM.score(ym_X_train, ym_y_train))

ym_pSVM = SVC(kernel='poly')
ym_pSVM.fit(ym_X_train, ym_y_train)
ym_pSVM_p = ym_pSVM.predict(ym_X_train)
print(ym_pSVM.score(ym_X_train, ym_y_train))

ym_rSVM = SVC(kernel='rbf')
ym_rSVM.fit(ym_X_train, ym_y_train)
ym_rSVM_p = ym_rSVM.predict(ym_X_train)
print(ym_rSVM.score(ym_X_train, ym_y_train))

od_mNB = MultinomialNB()
od_mNB.fit(od_X_train, od_y_train)
od_mNB_p = od_mNB.predict(od_X_train)
print(od_mNB.score(od_X_train, od_y_train))

od_cNB = CategoricalNB(min_categories=2)
od_cNB.fit(od_X_train, od_y_train)
od_cNB_p = od_cNB.predict(od_X_train)
print(od_cNB.score(od_X_train, od_y_train))

od_gNB = GaussianNB()
od_gNB.fit(od_X_train, od_y_train)
od_gNB_p = od_gNB.predict(od_X_train)
print(od_gNB.score(od_X_train, od_y_train))

od_lSVM = SVC(kernel='linear')
od_lSVM.fit(od_X_train, od_y_train)
od_lSVM_p = od_lSVM.predict(od_X_train)
print(od_lSVM.score(od_X_train, od_y_train))

od_pSVM = SVC(kernel='poly')
od_pSVM.fit(od_X_train, od_y_train)
od_pSVM_p = od_pSVM.predict(od_X_train)
print(od_pSVM.score(od_X_train, od_y_train))

od_rSVM = SVC(kernel='rbf')
od_rSVM.fit(od_X_train, od_y_train)
od_rSVM_p = od_rSVM.predict(od_X_train)
print(od_rSVM.score(od_X_train, od_y_train))

print("END")
