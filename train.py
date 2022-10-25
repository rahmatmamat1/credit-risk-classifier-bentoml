import pandas as pd
import numpy as np
import bentoml

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

# Read data
data = 'CreditScoring.csv'
df = pd.read_csv(data)

# Preprocessing
# lower column names
df.columns = df.columns.str.lower()

# Mapping value
status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}
df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}
df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}
df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}
df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}
df.job = df.job.map(job_values)

# Missing value handling
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

df = df[df.status != 'unk'].reset_index(drop=True)

# Split train test dataset
df_train, df_test = train_test_split(df, test_size=0.2, random_state=11)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = (df_train.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

del df_train['status']
del df_test['status']

# Features encoding
dv = DictVectorizer(sparse=False)

train_dicts = df_train.fillna(0).to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test.fillna(0).to_dict(orient='records')
X_test = dv.transform(test_dicts)


# ### Random forest

# rf = RandomForestClassifier(n_estimators=200,
#                             max_depth=10,
#                             min_samples_leaf=3,
#                             random_state=1)
# rf.fit(X_train, y_train)


# ### XGBoost
# 
# Note:
# 
# We removed feature names
# 
# It was 
# 
# ```python
# features = dv.get_feature_names_out()
# dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
# ```
# 
# Now it's
# 
# ```python
# dtrain = xgb.DMatrix(X_train, label=y_train)
# ```


# Create train matrix
dtrain = xgb.DMatrix(X_train, label=y_train)

# Boosting tree parameters
xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Train gradient boosting tree
model = xgb.train(xgb_params, dtrain, num_boost_round=175)


# BentoML

bentoml.xgboost.save_model(
    'credit_risk_model',
    model,
    custom_objects={
        'dictVectorizer': dv
    },
    signatures={
        "predict": {
            "batchable": True,
            "batch_dim": 0,
        }
    }
    )




