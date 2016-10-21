import pandas as pd
import numpy as np
import SFlibfm
import seaborn as sns
import time
from aa.utils import load_dataframe, store_dataframe
from sklearn import metrics
from sklearn.feature_extraction import DictVectorizer

df_train = load_dataframe('test', 'svd_train_df_week')
df_test = load_dataframe('test', 'svd_test_df_week')

# Convert client and style id to categorical variables
df_train['client_id'] = df_train['client_id'].map(lambda x : str(x))
df_train['style_id'] = df_train['style_id'].map(lambda x : str(x))
df_test['client_id'] = df_test['client_id'].map(lambda x : str(x))
df_test['style_id'] = df_test['style_id'].map(lambda x : str(x))

cols = ["client_id", "style_id"]
data_train = df_train[cols]
data_test = df_test[cols]
y_train = (np.array(df_train['sold']) - 0.5) * 2
y_test = (np.array(df_test['sold']) - 0.5) * 2

dictVec = DictVectorizer()
X_train_dict = data_train.T.to_dict().values()
X_test_dict = data_test.T.to_dict().values()
X_train = dictVec.fit_transform(X_train_dict)
X_test = dictVec.transform(X_test_dict)

# Build and train a Factorization Machine with adaptive SGD
start = time.time()
fm = SFlibFM.FM(num_factors= 2, num_iter = 10, task = "classification", initial_learning_rate = 0.01, 
  learning_rate_schedule = "constant", validation_size = 0.05, init_stdev = 0.1)
fm.fit(X_train, y_train, dictVec)
done = time.time()
print(done - start)

# Evaluate
preds = fm.predict(X_test)
auc = metrics.roc_auc_score(y_test, preds)
print round(auc, 3)

# Extract coefficients
coef_json = fm.extract_toJSON(dictVec)
store_dataframe(coef_json, 'test', 'sample_docker_fm')