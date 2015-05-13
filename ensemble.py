""" Amazon Access Challenge Starter Code
This code enhances the original categorical data with a few sets of additional
features.  It then trains ensemble of decision tree models on the data.

Hyperparameter searches were done using leaderboard feedback and grid searches
in earlier versions of this code.

Model blending / stacking was done in coordination with my partner, Paul Duan,
and his models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn import (metrics, cross_validation, linear_model, preprocessing)

SEED = 42  # a seed for randomized procedures

# === load data in memory === #
"""  Reading data. Each file has 10 columns.  The first column in train is the 
label (ACTION), the first column in test is an ID used for making sumbissions.
The last column, ROLE_CODE, is actually just a duplicate of another column.
"""
print "loading data"
test = pd.read_csv('test.csv', index_col=0)
train = pd.read_csv('train.csv') # There's no id column in train
y = train['ACTION']
train = train.drop(['ACTION'], axis=1)

"""I believe I got these hyper-parameters from a combination of leaderboard
feedback and an earlier version of this program that performed a somewhat
crude grid-search CV. This code is from one of my earliest Kaggle contests and
one of my earliest machine_learning methodlogy-based projects.  My
methodology has improved and streamlined a lot since then."""
modelRF =RandomForestClassifier(n_estimators=999, max_features='sqrt', 
		max_depth=None, min_samples_split=9, random_state=SEED)
modelXT =ExtraTreesClassifier(n_estimators=999, max_features='sqrt', 
		max_depth=None, min_samples_split=8, random_state=SEED)
modelGB =GradientBoostingClassifier(n_estimators=50, learning_rate=0.20, 
		max_depth=20, min_samples_split=9, random_state=SEED)

# Put train and test together for consistant preprocessing
X_all = pd.concat([test, train], ignore_index=True)
test_rows = len(test)

# This column is completely redundant with ROLE_TITLE
X_all = X_all.drop(['ROLE_CODE'], axis=1)

"""The feature "Role Title" is a subcategory of "Role Family" and Rollup 1 is a
subcategory of Rollup 2. I believe I experimented with leaving the redundant 
category as a feature and also with simply dropping it. 

But in the end I found a slight score boost from the code below, which preserves
a small amount of numerical association between subcategories with a common
super-category."""
X_all['ROLE_TITLE'] = X_all['ROLE_TITLE'] + (1000 * X_all['ROLE_FAMILY'])
X_all['ROLE_ROLLUPS'] = X_all['ROLE_ROLLUP_1'] + (10000 * X_all['ROLE_ROLLUP_2'])
X_all = X_all.drop(['ROLE_ROLLUP_1','ROLE_ROLLUP_2','ROLE_FAMILY'], axis=1)


# Adding Count Features for Each Column
print "Counts"
for col in X_all.columns:
    count = X_all[col].value_counts()
    X_all['count_'+col] = X_all[col].replace(count)

"""Resource is a bit different from the other columns. Others describe a 
particular department or role (like accountant, programmer, etc.). Resource is 
something that the employee wants access too.  In the following code, I measure
the % of the requests that each department, role, manager, etc. give for this
resource.  So in other words is this a resource that makes up a significant
fraction of the requests associated with this department?"""
# Adding features describing % of requests that are for the requested resource
# for each particular category of the other columns.
# This takes quite a while to run (10 mins on my machine.)
for col in X_all.columns[1:6]:
    X_all['resource_usage_'+col] = 0.0
    counts = X_all.groupby([col, 'RESOURCE']).size()
    percents =  counts.groupby(level=0).transform(lambda x: x/sum(x))
    cc = 0
    print col, len(percents)
    for c, r in percents.index:
        X_all.loc[(X_all[col]==c) & (X_all['RESOURCE']==r), 'resource_usage_'+ col] = percents[(c, r)]
        cc += 1
        if cc % 1000 == 1:
            print cc


# Number of Resources that a manager manages. I recall that many other similar
# features were tested, but this is the only that seemed to reliably move the 
# needle.
m_r_counts = X_all.groupby(['MGR_ID', "RESOURCE"]).size()
m_counts = m_r_counts.groupby(level=0).size()
X_all['Manager_Resrouces'] = X_all['MGR_ID'].replace(m_counts)


# Here running Pickle or cPickle would probably be helpful, depending
# on the workflow and goals at this stage of the competition


# Recover Test/Train
X_train = X_all.iloc[test_rows:,:]
X_test = X_all.iloc[:test_rows:]

# === Predictions === #
modelRF.fit(X_train, y)
modelXT.fit(X_train, y)
modelGB.fit(X_train, y)

predsRF = modelRF.predict_proba(X_test)[:, 1]
predsXT = modelXT.predict_proba(X_test)[:, 1]
predsGB = modelGB.predict_proba(X_test)[:, 1]

preds = np.vstack((predsRF, predsXT, predsGB)).T
submissions = pd.DataFrame(data=preds, columns=["RF", "XT", "GB"], index = test.index)
print submissions.describe()

# At this point these models were blended with my logistic model and another
# dozen or so models created by my competition partner.
# I think we did this together, in an interactive session using a standard
# stacking / blending techniques.
submissions.to_csv("dense_numerical_and_categorical_models.csv")
