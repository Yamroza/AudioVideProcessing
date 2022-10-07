# Laboratory 2. of Audio Processing, Video Processing and Computer Vision
# Scale-Space Blob Detectors
# Authors: 
# - David Štych
# - Aleksandra Jamróz


import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import io,color,img_as_float,transform
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy import ndimage
import numpy as np
import time
import math

#SELECT INPUT FILE HERE
FILENAME='fruits.jpg'
FILENAME='water_texture.jpg'
FILENAME='sunflowers.jpg'

SIGMA_O = 2/1.414

#PARAMETERS
if FILENAME=='fruits.jpg':
    THRESHOLD_ABS=0.0005
    THRESHOLD_REL=0.5
    STEPS=9
    SIGMA_RATIO = 1.4


if FILENAME=='water_texture.jpg':
    THRESHOLD_ABS=0.01
    THRESHOLD_REL=0.54
    STEPS=8
    SIGMA_RATIO = 1.4


if FILENAME=='sunflowers.jpg':
    THRESHOLD_ABS=0.0
    THRESHOLD_REL=0.45
    STEPS=8
    SIGMA_RATIO = 1.6


#REMOVE OVERLAPPING CIRCLES
def prune_circles(circles,img):
    to_remove=[]
    for i in range(len(circles[0])):
    
        if i in to_remove:
            continue
        cx1=circles[0][i]
        cy1=circles[1][i]
        rad1=circles[2][i]
        for k in range(i+1,len(circles[0])): 
            
            new_cord=[[],[],[]]
            
            if k in to_remove:
                continue

            cx2=circles[0][k]
            cy2=circles[1][k]
            rad2=circles[2][k]
            if math.sqrt((cx1-cx2)**2+(cy1-cy2)**2)<(rad2+rad1):
                to_remove.append(i)
                break


    circles[0]=np.delete(circles[0],to_remove)
    circles[1]=np.delete(circles[1],to_remove)
    circles[2]=np.delete(circles[2],to_remove)
    
    return circles


def show_all_circles(image, cx, cy, rad, color='r'):
    """
    image: numpy array, representing the grayscsale image
    cx, cy: numpy arrays or lists, centers of the detected blobs
    rad: numpy array or list, radius of the detected blobs
    """
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.imshow(image, cmap='gray')
    for x, y, r in zip(cx, cy, rad):
        circ = Circle((x, y), r, color=color, fill=False)
        ax.add_patch(circ)

    plt.title('%i circles' % len(cx))
    plt.show()


#FIND BLOBS BY CHANGING FILTER SIZE
def find_circles_changing_filter(image,scales):

    coordinates=[[],[],[]]
    for s in scales:
        if FILENAME=='fruits.jpg':
            filtered = -ndimage.gaussian_laplace(image, sigma=s)
        elif FILENAME=='water_texture.jpg': 
            filtered = ndimage.gaussian_laplace(image, sigma=s)
        else:
            filtered = ndimage.gaussian_laplace(image, sigma=s)**2
        filtered *= s**2
        
        #FIND PEAKS
        peaks = peak_local_max(
            filtered,
            threshold_abs=THRESHOLD_ABS,
            threshold_rel=THRESHOLD_REL,
            footprint=np.ones((3,3))
        )

        if len(peaks[:,0])>0:
            coordinates[0]=np.concatenate([coordinates[0],peaks[:,0]])
            coordinates[1]=np.concatenate([coordinates[1],peaks[:,1]])
            coordinates[2]=np.concatenate([coordinates[2],[s*1.414 for x in range(len(peaks[:,0]))]])

    return coordinates    
    
    
#FIND BLOBS BY CHANGING IMAGE SIZE
def find_circles_changing_image(original_image,scales):

    coordinates=[[],[],[]]

    for k in range(len(scales)): 

        rescale_factor=1.0/SIGMA_RATIO**k
        new_size=(int(original_image.shape[0]*rescale_factor),int(original_image.shape[1]*rescale_factor))
        image=transform.resize(original_image, new_size)

        if FILENAME=='fruits.jpg':
            filtered = -ndimage.gaussian_laplace(image, sigma=SIGMA_O)
        elif FILENAME=='water_texture.jpg': 
            filtered = ndimage.gaussian_laplace(image, sigma=SIGMA_O)
        else:
            filtered = ndimage.gaussian_laplace(image, sigma=SIGMA_O)**2
        
        filtered=transform.resize(filtered, original_image.shape)
        
            
        peaks = peak_local_max(
            filtered,
            threshold_abs=THRESHOLD_ABS,
            threshold_rel=THRESHOLD_REL,
            footprint=np.ones((3,3))
        )
        

        if len(peaks[:,0])>0:
            coordinates[0]=np.concatenate([coordinates[0],peaks[:,0]])
            coordinates[1]=np.concatenate([coordinates[1],peaks[:,1]])
            coordinates[2]=np.concatenate([coordinates[2],[SIGMA_O*SIGMA_RATIO**k*1.414 for x in range(len(peaks[:,0]))]])

    return coordinates    

    

original_image = io.imread(FILENAME)
if len(original_image.shape) > 2:
    gray_image = img_as_float(color.rgb2gray(original_image))
else:
    gray_image = img_as_float(original_image)

scales = np.array([SIGMA_O * (SIGMA_RATIO ** i) for i in range(STEPS)])


start = time.time()
coordinates = find_circles_changing_filter(gray_image,scales)
if FILENAME != 'sunflowers.jpg':
    coordinates = prune_circles(coordinates,original_image)
end = time.time()
print("INFO: Operation by changing the image filter took: %.3f ms" %  ((end - start)*1000))
show_all_circles(original_image, coordinates[1], coordinates[0],coordinates[2])



start = time.time()
coordinates = find_circles_changing_image(gray_image,scales)
if FILENAME != 'sunflowers.jpg':
    coordinates = prune_circles(coordinates,original_image)
end = time.time()
print("INFO: Operation by changing the image size took: %.3f ms" %  ((end - start)*1000))
show_all_circles(original_image, coordinates[1], coordinates[0],coordinates[2])


        


