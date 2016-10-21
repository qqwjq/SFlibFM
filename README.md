# Factorization Machine

This is a python implementation of Factorization Machines [1]. This uses stochastic gradient descent with adaptive regularization as a learning method, which adapts the regularization automatically while training the model parameters. See [2] for details. From libfm.org: "Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain."

[1] Steffen Rendle (2012): Factorization Machines with libFM, in ACM Trans. Intell. Syst. Technol., 3(3), May.

[2] Steffen Rendle: Learning recommender systems with adaptive regularization. WSDM 2012: 133-142

## Installation
```
python setup.py build_ext --inplace && mkdir SFlibFM && sudo pip install -e .
```

## Dependencies
* numpy
* sklearn

## Training Representation
The easiest way to use this class is to represent your training data as lists of standard Python dict objects, where the dict elements map each instance's categorical and real valued variables to its values. Then use a [sklearn DictVectorizer](http://scikit-learn.org/dev/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer) to convert them to a design matrix with a one-of-K or “one-hot” coding.

## Getting Started
Here's an example on one week of stitch fix shipment data to demonstrate how to use FM to estimate latent factor model.

```python
import pandas as pd
import numpy as np
import SFlibFM
import seaborn as sns
import time
from aa.utils import load_dataframe
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
round(auc, 3)

# Extract coefficients
coef_json = fm.extract_toJSON(dictVec)
coef_pandas = fm.extract_toPandas(dictVec)
sns.violinplot(coef_pandas.v0, groupby=coef_pandas.Attribute_name,
               order = cols)

# Writing JSON data
with open('fm_results.json', 'w') as f:
     f.write(json.dumps(result_json, sort_keys=True, indent=4, separators=(',', ': ')))
```