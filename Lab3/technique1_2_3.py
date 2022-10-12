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
from skimage.filters import threshold_otsu, gaussian, median, sobel
from skimage.util import img_as_ubyte
from skimage.color import label2rgb, gray2rgb
from skimage.segmentation import  watershed, expand_labels
import cv2
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk


#TOREMOVE
#process only n images to speed up testing
N_IMAGES=40
#number of worst images to show
N_IMGS_TO_SHOW=5

# -----------------------------------------------------------------------------
#
#     FUNCTIONS
#
# -----------------------------------------------------------------------------

# ----------------------------- Preprocess function -------------------------
def preprocess(image):
    image = median(image)
    image = exposure.equalize_adapthist(image)
    return image


# ----------------------------- Postprocess function -------------------------
def postprocess(image):
    footprint = disk(6)
    
    # works cool:
    image = dilation(image, footprint)
    image = erosion(image, footprint)

    # # meh:
    # image = closing(image, footprint) 
    # image = opening(image, footprint)

    # # even worse:
    # image = opening(image, footprint)
    # image = closing(image, footprint)

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
    #image = gaussian(image, 2)

    #PREPROCESSING
    image=preprocess(image)
    
    #otsu_th = threshold_otsu(image)
    #predicted_mask = (image > otsu_th).astype('int')

    edges = sobel(image)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < 0.11] = background
    markers[image > 0.55] = foreground

    ws = watershed(edges, markers)
    predicted_mask = label(ws == foreground)
    predicted_mask = expand_labels(predicted_mask, distance=2)
    predicted_mask = postprocess(predicted_mask)

    """
    plt.imshow(label2rgb(predicted_mask, image=image, bg_label=0), cmap='gray')
    plt.show()
    print(predicted_mask)
    print(np.max(image))
    print(np.mean(image))
    print(np.median(image))



    plt.imshow(image, cmap='gray')
    plt.show()
    h, bins = exposure.histogram(image)
    plt.figure()
    plt.plot(bins,h)
    plt.show()
    plt.imshow(predicted_mask, cmap='gray')
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
            alpha=0.4

            comparison_img = gray2rgb(preprocessed_img) * (1.0 - alpha) + comparison_img * alpha
            fig, axs = plt.subplots(2, 2)
            fig.suptitle('Worst images')

            axs[0,0].imshow(preprocessed_img, cmap='gray')
            axs[0, 0].set_title('Preprocessed image')
            axs[0,1].imshow(my_mask, cmap='gray')
            axs[0, 1].set_title('Predicted mask')
            axs[1,0].imshow(gt_mask, cmap='gray')
            axs[1, 0].set_title('Gt mask')
            axs[1,1].imshow(comparison_img)
            axs[1, 1].set_title('Comparison')
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
# path_im='reduced_subset/rawimages'
# path_gt='reduced_subset/groundtruth'
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
