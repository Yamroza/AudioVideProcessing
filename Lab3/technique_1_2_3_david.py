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
from skimage import io,exposure, morphology
from sklearn.metrics import jaccard_score
from skimage.util import img_as_float
from skimage.measure import label
from skimage.filters import threshold_otsu, gaussian, median, sobel
from skimage.util import img_as_ubyte
from skimage.color import label2rgb, gray2rgb
from skimage.segmentation import  watershed,expand_labels
from scipy import ndimage as ndi
from skimage.morphology import erosion, dilation, opening, closing,disk

from skimage.feature import peak_local_max
import cv2


#TOREMOVE
#process only n images to speed up testing
N_IMAGES=50
#number of worst images to show
N_IMGS_TO_SHOW=5

# -----------------------------------------------------------------------------
#
#     FUNCTIONS
#
# -----------------------------------------------------------------------------
# ----------------------------- Preprocess function ---------------------------
def preprocess(image):
    image = median(image)
    image=exposure.equalize_adapthist(image)
    return image
    
# ----------------------------- Postprocess function -------------------------
def postprocess(image):
    footprint = disk(6)
    image = dilation(image, footprint)
    image = erosion(image, footprint)
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
    # Code for the BASELINE system
    # - - - IMPLEMENT HERE YOUR PROPOSED SYSTEM - - -
        
    image = io.imread(img_file)
    image = img_as_float(image)
    #image = gaussian(image, 2)
    #PREPROCESSING
    image=preprocess(image)
    

    
    otsu_th = threshold_otsu(image)
    mask = (image > otsu_th).astype('int')


    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)

    sure_bg = cv2.dilate(thresh,kernel,iterations=2)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thresh,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    markers = cv2.watershed(img,markers)
    
    border=cv2.subtract(sure_bg,sure_fg)
    markers[sure_bg==0]=0
    #markers[unknown==255]=0
    predicted_mask=markers



    predicted_mask = expand_labels(predicted_mask, distance=2)

    # Postprocessing
    predicted_mask = postprocess(predicted_mask)


    
    """
    plt.imshow(markers)
    plt.show()
    """
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
    #TOREMOVE
    masks=[]
    
    
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

    
        #TOREMOVE
        masks.append(predicted_mask)
        if (i+1)==N_IMAGES:
            break     
       
    #TOREMOVE - SHOW 5 WORST IMAGES
    """
    """
    if N_IMGS_TO_SHOW>0:
        print("INFO: Showing %d worst images..." % N_IMGS_TO_SHOW)
        for value in sorted(IoU[:N_IMAGES])[:N_IMGS_TO_SHOW]:
            index=list(IoU[:N_IMAGES]).index(value)
            print("Image IoU %.3f" % value)
            preprocessed_img=preprocess(img_as_float(io.imread(img_files[index])))
            my_mask=masks[index]
            gt_mask=io.imread(gt_mask_files[index])
            
            my_mask_cells=np.ones((my_mask.shape))*(my_mask>0)
            gt_mask_cells=np.ones((gt_mask.shape))*(gt_mask>0)
            comparison_img=np.zeros((my_mask.shape[0],my_mask.shape[1],3))
            comparison_img[:,:,0]=my_mask_cells!=gt_mask_cells
            comparison_img[:,:,1]=my_mask_cells==gt_mask_cells
            non_zero_regions=np.ones((my_mask.shape))*(my_mask+gt_mask>0) 
            comparison_img[:,:,0]=comparison_img[:,:,0]*non_zero_regions
            comparison_img[:,:,1]=comparison_img[:,:,1]*non_zero_regions
            alpha=0.4

            comparison_img = gray2rgb(preprocessed_img) * (1.0 - alpha) + comparison_img * alpha
            fig, axs = plt.subplots(2, 2)
            fig.suptitle('Worst images')

            axs[0,0].imshow(preprocessed_img, cmap='gray')
            axs[0, 0].set_title('Preprocessed image')
            axs[0,1].imshow(my_mask_cells)
            axs[0, 1].set_title('Predicted mask')
            axs[1,0].imshow(gt_mask_cells)
            axs[1, 0].set_title('Gt mask')
            axs[1,1].imshow(comparison_img)
            axs[1, 1].set_title('Comparison')
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
            plt.show()
            
            

    #TOREMOVE
    return np.average(IoU[:N_IMAGES])
    #return np.average(IoU)

plt.close('all')

# -----------------------------------------------------------------------------
#
#     READING IMAGES
#
# -----------------------------------------------------------------------------

data_dir= os.curdir
#path_im='reduced_subset/rawimages'
#path_gt='reduced_subset/groundtruth'
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
