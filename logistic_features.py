__author__ = 'Benjamin Solecki'
__email__ = 'bensolucky@gmail.com'
__date__ = '07-16-2013'

"""
This program is based on code submitted by Miroslaw Horbal to the Kaggle 
forums, which was itself based on an earlier submission from Paul Doan.
My thanks to both.
"""

from numpy import array, hstack
from sklearn import metrics, cross_validation, linear_model, naive_bayes
from sklearn import preprocessing
from scipy import sparse
from itertools import combinations

import numpy as np
import pandas as pd
SEED = 42 # Set a random seed.

""" 
numpy.array -> numpy.array
    
Groups columns of data into all possible combinations of n degrees. 
This function comees almost directly from Miroslaw's original code but I added
a modification to remove redundant combinations. The function uses hashing
to assign a unique key to each possible combination between n categorical
columns.

Returns the array of combinations
"""
def group_data(data, degree=3, hash=hash):
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        # Column 7 is a subset of 5, so it redundant to combine them
	if 5 in indicies and 7 in indicies:
	    continue
        # Same here with column 2 and 3.
	elif 2 in indicies and 3 in indicies:
	    continue
	else:
            new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T


"""
This function also comes directly from Miroslaw's original code and the 
below comments are his own:

OneHotEncoder takes data matrix with categorical columns and
converts it to a sparse binary matrix.
     
Returns sparse binary matrix and keymap mapping categories to indicies.
If a keymap is supplied on input it will be used instead of creating one
and any categories appearing in the data that are not in the keymap are
ignored.
"""
def OneHotEncoder(data, keymap=None):

     if keymap is None:
          keymap = []
          for col in data.T:
               uniques = set(list(col))
               keymap.append(dict((key, i) for i, key in enumerate(uniques)))
     total_pts = data.shape[0]
     outdat = []
     for i, col in enumerate(data.T):
          km = keymap[i]
          num_labels = len(km)
          spmat = sparse.lil_matrix((total_pts, num_labels))
          for j, val in enumerate(col):
               if val in km:
                    spmat[j, km[val]] = 1
          outdat.append(spmat)
     outdat = sparse.hstack(outdat).tocsr()
     return outdat, keymap

""" This function is a modified version of the code that originally appeared in
Paul Duan's starter code from the forum. I believe it could be fairly easily
replaced by sklearn's cross_validation.cross_val_score function.  But in July
of 2013 I was not as familiar with sklearn and the function may not have even
existed then."""
def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
                                       X, y, test_size=1.0/float(N), 
                                       random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:,1]
        auc = metrics.roc_auc_score(y_cv, preds)
        mean_auc += auc
    return mean_auc/N


###############################################################################
# Code execution begins here
print "Reading dataset..."

"""  Reading data. Each file has 10 columns.  The first column in train is the 
label (ACTION), the first column in test is an ID used for making sumbissions.
The last column, ROLE_CODE, is actually just a duplicate of another column.
"""
test = pd.read_csv('test.csv', index_col=0)
train = pd.read_csv('train.csv') 
input_cols = train.columns[1:-1] # don't need first (label) or last (duplicate)
y = array(train.ACTION)
print "These are the inputs used:",
print input_cols
num_cols = len(input_cols)

""" Although I've read it in using pandas, this code doesn't really take 
advantage of much pandas functionality and from here treats the data as numpy
arrays and scipy sparse arrays """
all_data = np.vstack((train[input_cols].values, test[input_cols]))
train_rows = len(train)
print "Combined test, train data rows and columns:",
print all_data.shape

# Transform data
print
print "Transforming data..."

"""Here we relable the encoding for each category such that the possible values
range from 0 to number_of_categories-1. If I remember correctly, this makes the
hashing (the next step afterwards) better behaved. """

relabler = preprocessing.LabelEncoder()
for col in range(num_cols):
    all_data[:, col] = relabler.fit_transform(all_data[:, col])

"""These new arrays below use hashing to generate unique encodings for each
possible 2nd or 3rd order combination of the input columns. All possible does
not include some redundant combinations involving columns which are
subcategories of another function. """
db = group_data(all_data, degree=2) 
dt = group_data(all_data, degree=3)
print "1st order shape:", all_data.shape
print "2nd order shape:", db.shape
print "3rd order shape:", dt.shape


""" This function takes a numpy 2d array and operates on it column-wise.
The goals is take a column of categorical values and replace the rarest values
with some new constant, so that all "rare" values share a common value. Through
experimentation I found that setting values that appeared once or twice to 
their own values was optimal, although I experimented with 3, 4, etc..

The function makes use of numpy.bincount, which can only store up to 65,535
frequencies, and so for columns with more than those, I have to perform the 
calculation in a much slower way

This was was one of the strongest additions to the code and improved my 
leaderboard score to the top 10 or 20."""
def combine_rare_categories(combos):
    r, c = combos.shape
    print "Counts of unique values"
    print "Col	Before	After"
    for col in range(c):
        # Here we relable category encodings from 0 to number_categories-1
	# This allows better use of the bincount function below, since it
	# can only store up to a maximum of 65,534 bins.
        relabler = preprocessing.LabelEncoder()
        combos[:, col] = relabler.fit_transform(combos[:, col])
	# Uniques holds the number of unique values in this column and also
	# happens to be equal to the maximimum value + 1. Since this is the
	# case, we can use this value as the encoding number for very rare
	# count categories.
        uniques = len(np.unique(combos[:,col]))
        if uniques <= 65535:
            count_map = np.bincount(combos[:, col])#.astype('uint16'))
            for row_index, val in enumerate(combos[:, col]):
                if count_map[val] == 1:
                    combos[row_index, col] = uniques
	        elif count_map[val] == 2:
                    combos[row_index, col] = uniques + 1
        else: # 150 x slower
            print "A very slow column coming..."
            for row_index, val in enumerate(combos[:, col]):
                if (combos[:, col] == val).sum() == 1:
                    combos[row_index, col] = uniques
                elif (combos[:, col] == val).sum() == 2:
                    combos[row_index, col] = uniques + 1

        print col, "\t", uniques, "\t", len(np.unique(combos[:,col]))
        combos[:, col] = relabler.fit_transform(combos[:, col])
    return combos

print
print "Merging Rare Categories for the 3rd order combinations"
dt = combine_rare_categories(dt)
print
print "Merging Rare Categories for the 2nd order combinations"
db = combine_rare_categories(db)
print
print "Merging Rare Categories for the 1st order data (no combinations)"
all_data = combine_rare_categories(all_data)


# Collect the 1st, 2nd and 3rd order features together
X_all = np.hstack((all_data, db, dt))
num_features = X_all.shape[1]
print "Now", num_features, "categorical features"

X_train_all = X_all[:train_rows]
X_test_all = X_all[train_rows:]

# Define a model to train.  I also experimented with Naive Bayes, SVMs 
# and the SGDClassifier from sklearn on this data. Results were far and away
# best with Logistic, and I believe that other top competitors found the same.
model = linear_model.LogisticRegression(class_weight='auto', penalty='l2')
# Naive Bayes runs much, much faster but is less accurate. It's great for a 
# demo of the code but not for generating a high-scoring submission.
model = naive_bayes.BernoulliNB(alpha=0.03)

# Below is the one hot encoding that Miroslaw wrote. It appears that the 
# scklearn implementation does something a little differnt.
# Xts holds one hot encodings for each individual feature in memory,
# speeding up feature selection 
Xts = [OneHotEncoder(X_train_all[:,[i]])[0] for i in range(num_features)]

print
print "Performing greedy feature selection..."
""" Greedy feature selection loop.  This portion of code was provided by
Miroslaw to the forum's and, to my knowledge, nearly all of the top finishers
used some version of it to improve their scores.

It uses OneHotEncoding to binarize all categorical featuers into sparse
matrices.  However, most of 78 categorical features we have at this step have
thousands of individual categories, even after the merging of rare categories
I performed above.  So this loop does not perform the greedy feature selection
on all of these thousands of features.

Instead, it makes its feature selection and evaluation at the level of the the
78 categorical features. So at each step of the loop it adds all of the
thousands of binarized features from each column one at a time, then evaluates
performance with CV before deciding which of the 78 columns to add and when to
stop. """
score_hist = []
N = 10
good_features = set([])
while len(score_hist) < 2 or score_hist[-1][0] > score_hist[-2][0]:
    scores = []
    for f in range(len(Xts)):
        if f not in good_features:
            feats = list(good_features) + [f]
            Xt = sparse.hstack([Xts[j] for j in feats]).tocsr()
            score = cv_loop(Xt, y, model, N)
            scores.append((score, f))
            print "Feature: %i Mean AUC: %f" % (f, score)
    good_features.add(sorted(scores)[-1][1])
    score_hist.append(sorted(scores)[-1])
    print "Current features: %s" % sorted(list(good_features))
# Remove last added feature from good_features (I believe this is just a
# inconveniance arising from how this loop is set up).
good_features.remove(score_hist[-1][1])
good_features = sorted(list(good_features))
print len(good_features), " features"
# An example of good features found on a later run:
# good_features = [0,5,8,9,10,11,28,32,49,54,56,58,59,60,62,65,69,72,75]

print
print "Performing hyperparameter selection..."
""" This is another section where sklearn provides a very easy and simple
solution (gridSearchCV), but it either did not exist in July 2013 or I was not
aware of it at the time."""
score_hist = []
Xt = sparse.hstack([Xts[j] for j in good_features]).tocsr()
params = np.logspace(-4, 4, 15, base=2)  # I probably found this range through experimentation.
for C in params:
    model.alpha = C # For Naive Bayes
    model.C = C     # For Logistics
    score = cv_loop(Xt, y, model, N)
    score_hist.append((score,C))
    print "C: %f Mean AUC: %f" %(C, score)
bestC = sorted(score_hist)[-1][1]
print "Best C value: %f" % (bestC)
model.alpha = bestC
model.C = bestC
    
print "Performing One Hot Encoding on entire dataset (Using Only Good Features)"
Xt = np.vstack((X_train_all[:,good_features], X_test_all[:,good_features]))
Xt, keymap = OneHotEncoder(Xt)
X_train = Xt[:train_rows]
X_test = Xt[train_rows:]
    

print "Training full model..."
print "Making prediction and saving results..."
model.fit(X_train, y)
preds = model.predict_proba(X_test)[:,1]
submission = pd.Series(data=preds, index=test.index)
submission.to_csv("sparse_binary_model.csv")
