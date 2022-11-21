# -*- coding: utf-8 -*-
"""
CODE TO TUNE FEATURE EXTRACTION PARAMETERS

Audio features

"""


import template as 




##############################################################################
# Data read (and prepare)
##############################################################################

# Get file names
major_files = listdir('./data/Major')
minor_files = listdir('./data/Minor')
major_files = ['./data/Major/' + f for f in major_files]
minor_files = ['./data/Minor/' + f for f in minor_files]

# Unify data and code labels
X = deepcopy(minor_files)
X.extend(major_files)
# Label 0 --> Minor
# Label 1 --> Major (Arbitrary positive class)
y = list(np.concatenate((np.zeros(len(minor_files)), 
                         np.ones(len(major_files))), axis = 0).astype(int))

# Fix size of test set
test_size = 0.3
# Manual seed (for replicability)
ran_seed = 999
# Perform train-test division
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, 
                                                    random_state=ran_seed)

##############################################################################
# Training Process
##############################################################################

# Extract features for the training set
M_train = extract_features(X_train,verbose = False,flen=10,nsub=10,hop=10)   
# Normalize    
scaler = StandardScaler().fit(M_train)
M_train_n = scaler.transform(M_train)

# We use a Support Vector Machine with RBF kernel
clf = SVC(probability=True)
# Train model
clf.fit(M_train_n, y_train)

##############################################################################
# Evaluation Process
##############################################################################


# Extract features for the test set
M_test = extract_features(X_test)   
# Normalize
M_test_n = scaler.transform(M_test)

# Obtain predicted labels and scores (probabilities) according to model
y_pred = clf.predict(M_test_n)
y_scores = clf.predict_proba(M_test_n)[:,1]

# Obtain ROC curve values (FPR, TPR)
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
# Get Area Under the Curve
auc_svm = roc_auc_score(y_test, y_scores)
# Plot ROC curve (displaying AUC)
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic Curve - AUC = ' +
          str(np.round(auc_svm,3)))
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
