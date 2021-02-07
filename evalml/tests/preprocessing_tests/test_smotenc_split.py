# import numpy as np
# import pandas as pd
# import pytest
# from sklearn.model_selection import StratifiedKFold, train_test_split

# from evalml.preprocessing.data_splitters import SMOTENCCVSplit, SMOTENCTVSplit

# im = pytest.importorskip('imblearn.over_sampling', reason='Skipping plotting test because imblearn not installed')


# def test_smotenc_nsplits():
#     assert SMOTENCTVSplit().get_n_splits() == 1
#     assert SMOTENCCVSplit().get_n_splits() == 3
#     assert SMOTENCCVSplit(n_splits=5).get_n_splits() == 5


# @pytest.mark.parametrize("value", [np.nan, "hello"])
# @pytest.mark.parametrize("splitter", [SMOTENCTVSplit, SMOTENCCVSplit])
# def test_smotenc_error(splitter, value, X_y_binary):
#     X, y = X_y_binary
#     X = pd.DataFrame(X)
#     y = pd.Series(y)
#     X.iloc[0, :] = value
#     snc = splitter(categorical_features=[0])
#     with pytest.raises(ValueError, match="Values not all numeric or there are null values"):
#         # handles both TV and CV iterations
#         next(snc.split(X, y))
#     with pytest.raises(ValueError, match="Values not all numeric or there are null values"):
#         snc.transform(X, y)


# @pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
# @pytest.mark.parametrize('dataset', [0, 1])
# def test_smotenc_tv_default(data_type, make_data_type, dataset, X_y_binary, X_y_multi):
#     if dataset == 0:
#         X, y = X_y_binary
#     else:
#         X, y = X_y_multi
#     X = pd.DataFrame(X)
#     X['categorical_column'] = [i % 4 for i in range(len(X))]
#     X = X.values
#     cat_cols = [20]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#     st = im.SMOTENC(categorical_features=cat_cols, random_state=0)
#     X_resample, y_resample = st.fit_resample(X_train, y_train)
#     initial_results = [(X_resample, y_resample), (X_test, y_test)]
#     X_transform, y_transform = st.fit_resample(X, y)

#     X = make_data_type(data_type, X)
#     y = make_data_type(data_type, y)
#     stsplit = SMOTENCTVSplit(categorical_features=cat_cols, random_state=0, test_size=0.2)
#     for i, j in enumerate(stsplit.split(X, y)):
#         for idx, tup in enumerate(j):
#             for jdx, val in enumerate(tup):
#                 if jdx == 1:
#                     np.testing.assert_equal(val.to_series().values, initial_results[idx][jdx])
#                 else:
#                     np.testing.assert_equal(val.to_dataframe().values, initial_results[idx][jdx])

#     X_stsplit, y_stsplit = stsplit.transform(X, y)
#     np.testing.assert_equal(X_transform, X_stsplit.to_dataframe().values)
#     np.testing.assert_equal(y_transform, y_stsplit.to_series().values)


# @pytest.mark.parametrize('data_type', ['np', 'pd', 'ww'])
# @pytest.mark.parametrize('dataset', [0, 1])
# def test_smotenc_cv_default(data_type, make_data_type, dataset, X_y_binary, X_y_multi):
#     if dataset == 0:
#         X, y = X_y_binary
#     else:
#         X, y = X_y_multi
#     X = pd.DataFrame(X)
#     X['categorical_column'] = [i % 4 for i in range(len(X))]
#     X = X.values
#     cat_cols = [20]

#     skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=0)
#     st = im.SMOTENC(categorical_features=cat_cols, random_state=0)
#     initial_results = []
#     for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
#         X_resample, y_resample = st.fit_resample(X[train_indices], y[train_indices])
#         initial_results.append([(X_resample, y_resample), (X[test_indices], y[test_indices])])
#     X_transform, y_transform = st.fit_resample(X, y)

#     X = make_data_type(data_type, X)
#     y = make_data_type(data_type, y)
#     stsplit = SMOTENCCVSplit(categorical_features=cat_cols, random_state=0)
#     for i, j in enumerate(stsplit.split(X, y)):
#         for idx, tup in enumerate(j):
#             for jdx, val in enumerate(tup):
#                 if jdx == 1:
#                     np.testing.assert_equal(val.to_series().values, initial_results[i][idx][jdx])
#                 else:
#                     np.testing.assert_equal(val.to_dataframe().values, initial_results[i][idx][jdx])

#     X_stsplit, y_stsplit = stsplit.transform(X, y)
#     np.testing.assert_equal(X_transform, X_stsplit.to_dataframe().values)
#     np.testing.assert_equal(y_transform, y_stsplit.to_series().values)