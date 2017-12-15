"""
Author:     Mitchell Phillips
File:       slic_example.py
Date:       December 2017

Purpose: SLIC superpixel segmentation using OpenCV example. There 
exists plenty of scikit-image tutorials and examples, but not too 
many for OpenCV-python. This is a simple example on the 
implementation for SLIC. Here, a variation of SLIC, SLICO is used. 
"""

import cv2
import numpy as np

def slic_superpixels(img):
    """
    Perform SLIC superpixel segmentation with OpenCV. 
    INPUT:  img - Image to undergo SLIC segmentation.
    OUTPUT: [] - System out. SLIC segmented image saved to the 
            current working directory.
    """
    img_gauss = cv2.GaussianBlur(img,(5,5),0)

    slico = cv2.ximgproc.createSuperpixelSLIC(
        img_gauss, algorithm=cv2.ximgproc.SLICO, region_size=50)
    
    slico.iterate(10)
    slico.enforceLabelConnectivity()   
    mask = slico.getLabelContourMask()
    super_pixels = slico.getLabels()
    
    mask_ind = np.where(mask==-1)
    img[mask_ind] = np.array([0,255,255]) 
    cv2.imwrite('slic_img.jpg', img)


def main():
    img = cv2.imread('test_img.jpg')
    slic_superpixels(img)


if __name__ == "__main__":
    main()