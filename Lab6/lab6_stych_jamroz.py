# -*- coding: utf-8 -*-
"""
LAB TEMPLATE

Audio features

"""

# Generic imports
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
# Import ML tool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
# Import audio feature library
import librosa as lbs
# Import our custom lib for audio feature extraction (makes use of librosa)
import audio_features as af


NUMBER_OF_FEATURES=5
#If set to true, a function to find the parameters will be called. Results printed and script killed.
TUNE_PARAMETERS=False


##############################################################################
# Feature extraction function
##############################################################################

def extract_features(X,verbose = True,flen=1024,nsub=8,hop=256):
    """
    >> Function to be completed by the student
    Extracts a feature matrix for the input data X
    ARGUMENTS:
        X: Audio data file paths to analyze
        verbose: Verbose prints
        flen: Frame length (default in template.py was 512)
        nsub: Number of subframes (default in template.py was 10)
        hop: Hop length (default in template.py was 128)
    """
    # Number of samples to process
    num_data = len(X)
    # Sample rate of the signals    
    sr = lbs.get_samplerate(X[0])
    
    # Specify the number of features to extract
    n_feat = NUMBER_OF_FEATURES
    
    # Generate empty feature matrix
    M = np.zeros((num_data,n_feat))
    

    # Threshold below reference to consider as silence (dB) 
    thr_db = 20
    
    for i in range(num_data):
        if verbose:
            print('%d/%d... ' % (i+1, num_data), end='')
        # Read audio signal
        audio_data,_ = lbs.load(X[i], sr=sr)
        # Preprocessing (Trim + center)
        audio_data = af.preprocess_audio(audio_data, thr=thr_db)
        

        
        # Get first two features (Energy entropy mean and max)
        energy_entropies = af.get_energy_entropy(audio_data, 
                                                     flen=flen, 
                                                     hop=hop, 
                                                     nsub=nsub)
        # Compute mean (ignore nan values)
        M[i,0] = np.nanmean(energy_entropies)
        # Compute max value (ignore nan values)
        M[i,1] = np.nanmax(energy_entropies)

        #TODO
        ##########################################
        # Extract additional features
        
        #MEAN OF SPECTRAL ENTROPIES
        spectral_entropies=af.get_spectral_entropy(audio_data,flen=flen,hop=hop,nsub=nsub)                                           
        M[i,2] = np.nanmean(spectral_entropies)
        
        #MEAN OF SPECTRAL FLUX
        spectral_flux=af.get_spectral_flux(audio_data,flen=flen,hop=hop)                                           
        M[i,3] = np.nanmean(spectral_flux)
        
        #MEAN AND MAX OF ZERO CROSSING RATES
        zero_rates=af.get_zero_crossing_rate(audio_data, flen=flen, hop=hop)
        M[i,4] = np.nanmean(zero_rates)
        #This made it worse
        #M[i,5] = np.nanmax(zero_rates)
        
        
        #This made it worse too
        #MEAN OF SECTRAL CONTRAST
        #spectral_contrast=af.get_spectral_contrast(audio_data, flen=flen, hop=hop)
        #M[i,6] = np.nanmean(spectral_contrast)

        ##########################################
        
        if verbose:
            print('Done')
    return M

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
                                                    
#TODO
##############################################################################
# Tuning Process
##############################################################################    
if TUNE_PARAMETERS:
    #[flen,nsub,hop]
    parameter_list=[
        [512,10,128],
        [512,10,256],
        [512,8,128],
        [512,8,256],
        [1024,10,128],
        [1024,10,256],
        [1024,8,128],
        [1024,8,256]]
    scores=[]       
    for parameters in parameter_list:
        print("Testing parameters")
        print("flen =",parameters[0])
        print("nsub =",parameters[1])
        print("hop  =",parameters[2])
        #TRAIN MODEL
        M_train = extract_features(X_train,
                                    verbose=False,
                                    flen=parameters[0],
                                    nsub=parameters[1],
                                    hop=parameters[2])     
        scaler = StandardScaler().fit(M_train)
        M_train_n = scaler.transform(M_train)
        clf = SVC(probability=True)
        clf.fit(M_train_n, y_train)
        
        
        #TEST MODEL
        M_test = extract_features(X_test,
                                    verbose=False,
                                    flen=parameters[0],
                                    nsub=parameters[1],
                                    hop=parameters[2])    
        M_test_n = scaler.transform(M_test)
        y_pred = clf.predict(M_test_n)
        y_scores = clf.predict_proba(M_test_n)[:,1]
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_scores)
        auc_svm = roc_auc_score(y_test, y_scores)
        scores.append(auc_svm)
        print("AUC  =",auc_svm)
        print()
    max_value = max(scores)
    best_params=parameter_list[scores.index(max_value)]
    print(best_params)
    print()
    print("Parameters tuned. Maximum AUC =",max_value)
    print("flen =",best_params[0])
    print("nsub =",best_params[1])
    print("hop  =",best_params[2])
    #KILL THE SCRIPT
    import sys
    sys.exit()
                                    
##############################################################################
# Training Process
##############################################################################

# Extract features for the training set
M_train = extract_features(X_train,verbose=False)   
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
M_test = extract_features(X_test,verbose=False)   
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
