#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %      
%  Lab Session #3                                                       %
%                                                                       %
%  CELL SEGMENTATION                                                    %
%                                                                       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io,exposure
from sklearn.metrics import jaccard_score
from skimage.util import img_as_float
from skimage.measure import label
from skimage.filters import threshold_otsu, gaussian, median
import cv2

# -----------------------------------------------------------------------------
#
#     FUNCTIONS
#
# -----------------------------------------------------------------------------

# ----------------------------- Preprocess function -------------------------
def preprocess(image):
    image = median(image)
    # image = exposure.equalize_adapthist(image)
    return image


# ----------------------------- Segmentation function -------------------------
def cell_segmentation(img_file):
    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %                                                                       %
    %  Template for implementing the main function of the segmentation      %
    % system: 'cell_segmentation'                                           %
    % - Input: path to the image to be segmented                     %
    % - Output: predicted segmentation mask                                 %
    %                                                                       %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    CELL_SEGMENTATION:
    - - -  COMPLETE - - -
    """
        
    image = io.imread(img_file)
    image = img_as_float(image)

    #PREPROCESSING
    image = preprocess(image)

    otsu_th = threshold_otsu(image)
    predicted_mask = (image > otsu_th).astype('int')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    return predicted_mask
    
# ----------------------------- Evaluation function ---------------------------        
def evaluate_masks(img_files, gt_mask_files):
    """ EVALUATE_MASKS: 
        It receives two lists:
        1) A list of the paths to the images to be analysed
        2) A list of the paths to the corresponding Ground-Truth (GT) 
            segmentations
        For each iamge on the list:
            It perform the segmentation
            It determines Jaccard Index
        Finally, it computes an average Jaccard Index for all the images in 
        the list
    """
    def label2masks(label_image):
        M,N=label_image.shape
        Num_labels=np.max(label_image)
        masks=np.zeros((M,N,Num_labels),dtype=int)
        # print(f"Num_labels={Num_labels}")
        for l in range(1,Num_labels+1):
            masks[label_image==l,l-1]=1
        return masks
    
    # Getting the size of the images
    mask0=io.imread(gt_mask_files[0])
    M,N=mask0.shape
    # Downsampling factor to make the evaluation computationally efficient
    downs_f=0.1
    m=round(M*downs_f)
    n=round(N*downs_f)
    # Array containing the IoU associated with each image
    IoU=np.zeros(len(img_files))
    
    for i in range(len(img_files)):
        
        # Image segmentation
        predicted_mask = cell_segmentation(img_files[i])
        predicted_mask_label =label(predicted_mask)
        # Image downsampling for computational reasons
        predicted_mask_label = cv2.resize(predicted_mask_label, (n, m), interpolation = cv2.INTER_NEAREST)
        
        # Reading and downsampling ground-truth segmentations
        gt_label = io.imread(gt_mask_files[i])
        gt_label = cv2.resize(gt_label, (n, m), interpolation = cv2.INTER_NEAREST)
        
        # Converting the groundtruth label images in individual masks (one per cell)
        gt_masks=label2masks(gt_label)
        # Number of cells (groundtruth segmentation)
        Num_gt_cells=np.max(gt_label)
    
        # Converting the predicted label images in individual masks (one per object)
        pred_masks=label2masks(predicted_mask_label)
        # Number of predicted objects 
        Num_pred_cells=np.max(predicted_mask_label)
    
        # Computing the IoU of each cell in the groundtruth segmentation
        # (aproximate estimation)
        IoU_max=np.zeros((Num_gt_cells,2))
        for gt in range(Num_gt_cells):
            scores=np.zeros(Num_pred_cells)
            for pred in range(Num_pred_cells):
                if (np.sum(gt_masks[:,:,gt])==0 or np.sum(pred_masks[:,:,pred])==0):
                    # print("zero check!")
                    scores[pred]=0.0
                else: 
                    scores[pred]=jaccard_score(gt_masks[:,:,gt].flatten(), pred_masks[:,:,pred].flatten())
            
            # For each cell in the groudtruth segmentation,
            # IoU_max stores the maximum IoU achived after computing the IoU 
            # with respect to each object in the predicted segmentation
            IoU_max[gt,0]=np.max(scores)
            IoU_max[gt,1]=np.argmax(scores)
        
        # The IoU of each image is computed as the average of IoU of each groundtruth cell
        IoU[i]=np.average(IoU_max[:,0])
        print (f"Image {i}, IoU={IoU[i]}")

    
    
    return np.average(IoU)

plt.close('all')

# -----------------------------------------------------------------------------
#
#     READING IMAGES
#
# -----------------------------------------------------------------------------

data_dir= os.curdir
#path_im='reduced_subset/rawimages'
#path_gt='reduced_subset/groundtruth'
# path_im='subset/rawimages'
# path_gt='subset/groundtruth'
path_im='Lab3/subset/rawimages'
path_gt='Lab3/subset/groundtruth'

img_files = [ os.path.join(data_dir,path_im,f) for f in sorted(os.listdir(os.path.join(data_dir,path_im))) 
            if (os.path.isfile(os.path.join(data_dir,path_im,f)) and f.endswith('.tif')) ]

gt_mask_files = [ os.path.join(data_dir,path_gt,f) for f in sorted(os.listdir(os.path.join(data_dir,path_gt))) 
            if (os.path.isfile(os.path.join(data_dir,path_gt,f)) and f.endswith('.tif')) ]
# img_files.sort()
# gt_mask_files.sort()
print("Number of train images", len(img_files))
print("Number of image masks", len(gt_mask_files))

# -----------------------------------------------------------------------------
#
#     Segmentation and evaluation
#
# -----------------------------------------------------------------------------

mean_score = evaluate_masks(img_files, gt_mask_files)
print(f"Average IoU={mean_score}")
