import pandas as pd


def do_split(df, frTest):
    test = df.sample(frac=frTest, axis=0, replace=False)
    train = df.drop(index=test.index)
    return train, test


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
hk_Y_train = hk_raw_data_train[:, 0]
hk_X_test = hk_raw_data_test[:, 1:(hk_raw_data_test.shape[1])]
hk_Y_test = hk_raw_data_test[:, 0]

ym_raw_data_train = ym_train_df.to_numpy()
ym_raw_data_test = ym_test_df.to_numpy()
ym_X_train = ym_raw_data_train[:, 1:(ym_raw_data_train.shape[1])]
ym_Y_train = ym_raw_data_train[:, 0]
ym_X_test = ym_raw_data_test[:, 1:(ym_raw_data_test.shape[1])]
ym_Y_test = ym_raw_data_test[:, 0]

od_raw_data_train = od_train_df.to_numpy()
od_raw_data_test = od_test_df.to_numpy()
od_X_train = od_raw_data_train[:, 1:(od_raw_data_train.shape[1])]
od_Y_train = od_raw_data_train[:, 0]
od_X_test = od_raw_data_test[:, 1:(od_raw_data_test.shape[1])]
od_Y_test = od_raw_data_test[:, 0]

print("END")