This code produces a portion of our 1st place code from the Kaggle Amazon Access Competition.

My partner, Paul Duan, produced the other portion.  At the time we also had code to blend our various model outputs.

see: https://www.kaggle.com/c/amazon-employee-access-challenge

and: https://www.kaggle.com/c/amazon-employee-access-challenge/leaderboard/private

About the Data: 
The goal was to predict employee resource access grants from categorical job description data. The scoring metric is AUC (area under the ROC curve).  There are only 9 categorical input feature columns, one of which is completely redundant.  There are roughly 30,000 training rows and 50,000 test rows.

About the Code:
The general strategy was to produce 2 feature sets: one categorical to be modeled with decision tree based approaches and the second a sparse matrix of binary features, creataed by binarizing all categorical values and 2nd and 3rd order combinations of categorical values.  The starting point of this latter set of code was provided on the forums by Miroslaw Horbal.  The most critical modeification I made to it was in merging the mostly rarely occuring binary features into a much smaller number of features that held these rare values.

### Requirements: 
##### This code assumes you have the competition data (train.csv & test.csv) saved in the working directory.  
##### I have not provided the data set in this repository, it would need to be downloaded from the competition links to run the code fully.

It was run most recently on windows 8 with python 2.7.5 and the following package versions:

scikit-learn 14.1

pandas 12.0

numpy 1.8
