"""
**********************************************************************
Disclaimer: This script is a modification of Pathak's nlc.py script.
This implementation is intended to provide further clarification 
and provide information referring back to the journal articles where 
the algorithm originated from. Furthermore, portions of the script 
have been slimmed down or modified to utilize the opencv library 
rather than skimage.

Source:     https://github.com/pathak22/videoseg 


**********************************************************************

Author:     Mitchell Phillips
File:       iss_uNLC.py
Date:       December 2017
Purpose:    Provide clarification on the methods and functions used
for the uNLC algorithm. Portions of the algorithm will be refereed 
back to the corresponding journal article. As Pathak's uNLC is an 
adaptation of NLC, the majority of the algorithm details can be 
found in Faktor and Irani's paper.

References: 
[1] - Pathak et al., Learning Features by Watching Objects Move, 2017.
[2] - Faktor and Irani, Video Segmentation by Non-Local Consensus 
Voting, 2014

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import os
import sys

from PIL import Image
from skimage.segmentation import slic
from skimage.feature import hog
from skimage import color

from scipy.spatial import KDTree
from scipy.misc import imresize
from scipy import ndimage
from scipy.signal import convolve2d

import time
import utils
import _init_paths  # noqa

from mr_saliency import MR
import pyflow



def my_accumarray(indices, vals, size, func='plus', fill_value=0):
    """
    Implementing python equivalent of matlab accumarray.
    Taken from SDS repo: master/superpixel_representation.py#L36-L46
        indices: must be a numpy array (any shape)
        vals: numpy array of same shape as indices or a scalar
    """

    # get dictionary
    function_name_dict = {
        'plus': (np.add, 0.),
        'minus': (np.subtract, 0.),
        'times': (np.multiply, 1.),
        'max': (np.maximum, -np.inf),
        'min': (np.minimum, np.inf),
        'and': (np.logical_and, True),
        'or': (np.logical_or, False)}

    if func not in function_name_dict:
        raise KeyError('Function name not defined for accumarray')
    if np.isscalar(vals):
        if isinstance(indices, tuple):
            shape = indices[0].shape
        else:
            shape = indices.shape
        vals = np.tile(vals, shape)

    # get the function and the default value
    (function, value) = function_name_dict[func]

    # create an array to hold things
    output = np.ndarray(size)
    output[:] = value
    function.at(output, indices, vals)

    # also check whether indices have been used or not
    isthere = np.ndarray(size, 'bool')
    istherevals = np.ones(vals.shape, 'bool')
    (function, value) = function_name_dict['or']
    isthere[:] = value
    function.at(isthere, indices, istherevals)

    # fill things that were not used with fill value
    output[np.invert(isthere)] = fill_value
    
    return output



def region_extraction(img_sequence, maxsp=200, vis=False, redirect=False):
    """
    Region extraction process. uNLC obtains regions through SLIC 
    superpixel segmentation. This stage in the algorithm is where 
    uNLC and NLC differ from one another. Where NLC adopts a trained 
    edge detector, uNLC instead performs SLIC. While no parameters 
    were stated in Pathak's paper, the full pipeline uses a maximum
    superpixel count of 400. However, OpenCV's SLIC uses a maximum 
    region size per superpixel. The original region extraction 
    process can be found in section 3.2, 'Detailed Description of the
    Algorithm' under 'Region Extraction' [2]. For uNLC, this is 
    described in section 5.1, 'Unsupervised Motion Segmentation' [1].

    INPUT:  img_sequence - Image sequence undergoing uNLC.
    OUTPUT: superpixels - Superpixel segmentation labels for each 
            frame in a given image sequence.  
    """
    
    start_time = time.time()
    
    if img_sequence.ndim < 4:
        img_sequence = img_sequence[None, ...]

    superpixels = np.zeros(img_sequence.shape[:3], dtype=np.int)
    
    for i in range(img.shape[0]):

        # Obtain SLIC superpixel segmentation labels. 
        img_gauss = cv2.GaussianBlur(img_sequence[i],(5,5),0)
        slico = cv2.ximgproc.createSuperpixelSLIC(
            img_gauss, algorithm=cv2.ximgproc.SLICO, region_size=50)
        slico.iterate(10)
        slico.enforceLabelConnectivity()
        superpixels[i] = slico.getLabels()

        # Visualize the SLIC superpixel segmentation results. This 
        # needs more work in the current implementation. 
        if vis and False:
            mask = slico.getLabelContourMask()    
            mask_ind = np.where(mask==-1)
            img[mask_ind] = np.array([0,255,255]) 
            cv2.imwrite('slic_img.jpg', img)
        
        if not redirect:
            sys.stdout.write('Superpixel computation: [% 5.1f%%]\r' %(
                100.0 * float((i + 1) / img_sequence.shape[0])))
            sys.stdout.flush()

    end_time = time.time()
    print('Superpixel computation finished: %.2f s' % (end_time - start_time))

    if img_sequence.ndim < 4:
        return superpixels[0]
    
    return superpixels
    

def get_region_boxes(superpixels):
    """
    Get bounding boxes for each superpixel region. 
    INPUT: 
        superpixels - (h,w): 0-indexed regions, number of regions <= 
        max number of superpixels.
    OUTPUT: 
        boxes - (number of superpixels, 4) : (xmin, ymin, xmax, ymax)
    """
    x = np.arange(0, sp.shape[1])
    y = np.arange(0, sp.shape[0])
    xv, yv = np.meshgrid(x, y)
    sizeOut = np.max(sp) + 1
    sp1 = superpixels.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    spxmin = utils.my_accumarray(sp1, xv, sizeOut, 'min')
    spymin = utils.my_accumarray(sp1, yv, sizeOut, 'min')
    spxmax = utils.my_accumarray(sp1, xv, sizeOut, 'max')
    spymax = utils.my_accumarray(sp1, yv, sizeOut, 'max')

    boxes = np.hstack((spxmin.reshape(-1, 1), spymin.reshape(-1, 1),
                        spxmax.reshape(-1, 1), spymax.reshape(-1, 1)))
    return boxes


def color_histogram_descriptor(im, colBins):
    """
    Concatenation of RGB and LAB color histograms. This is one of the 
    measurements used for the Region Descriptor when computing the 
    nearest neighbor search. Each of the color histogram are 
    described by 20 bins. Thus, there are 6 * 20 bins in total, 
    covering both of the color spaces. There is a brief mention of 
    this process in [1], however, the details are explained in
    section 3.2, 'Detailed Description of the Algorithm - Region 
    Descriptor' [2].

    Note, in RGB colorspace, the channels range from 0 to 255.
    In LAB colorspace, the lightness value, L, ranges from 0 to 100 
    and the color-opponent values, a and b, range from -128 to 127.

    Input: im: (h,w,c): 0-255: np.uint8: RGB
    Output: descriptor: (colBins*6,)
    """
    
    assert im.ndim == 3 and im.shape[2] == 3, 'Requires RBG image.'
    
    rgb_lab = np.concatenate((im, color.rgb2lab(im)), axis=2).reshape((-1, 6))
    color_descriptor = np.zeros((colBins * 6,), dtype=np.float)
    
    histogram_range = (
        (0, 255),(0, 255),(0, 255),
        (0, 100),(-128, 127),(-128, 127))

    for i in range(6):
        
        color_descriptor[i*colBins:(i + 1)*colBins], _ = np.histogram(
            rgb_lab[:, i], bins=colBins, range=histogram_range[i])
        
        color_descriptor[i * colBins:(i + 1) * colBins] /= (
            np.sum(color_descriptor[i * colBins:(i + 1) * colBins]) 
            + (np.sum(color_descriptor[i * colBins:(i + 1) * colBins]) < 1e-4))
    
    return color_descriptor


def region_descriptor(
    im, sp, spPatch=15, colBins=20, hogCells=9,hogBins=6, redirect=False):
    """
    Compute region descriptors for NLC.
    Region descriptors are computed for each superpixel. 
    The descriptor is made up of a concatenation of four 
    measurements: RGB color histogram and LAB color histogram (6 
    channels, 20 bins each), histogram of oriented gradients 
    (9 cells, 6 orientation bins, computed over a 15x15 patch around 
    the superpixel) and the relative spatial coordinates of the 
    superpixel. See section 3.2, 'Detailed Description of the 
    Algorithm - Region Descriptor' [2].
    
    INPUT:  im - Image or image sequence.
            sp - Superpixel region labels. 
            im: (h,w,c) or (n,h,w,c): 0-255: np.uint8: RGB
            sp: (h,w) or (n,h,w): 0-indexed regions, #regions <= numsp
            spPatch: patchsize around superpixel for feature 
            computation
    
    OUTPUT: regions: (k,d) where k < numsp*n
            frameEnd: (n,): indices of regions where frame ends: 
            0-indexed, included   
    """

    sTime = time.time()
    
    if im.ndim < 4:
        im = im[None, ...]
        sp = sp[None, ...]

    hogCellSize = int(spPatch / np.sqrt(hogCells))
    n, h, w, c = im.shape
    d = 6 * colBins + hogCells * hogBins + 2
    numsp = np.max(sp) + 1  # because sp are 0-indexed
    regions = np.ones((numsp * n, d), dtype=np.float) * -1e6
    frameEnd = np.zeros((n,), dtype=np.int)
    count = 0
    for i in range(n):
        boxes = get_region_boxes(sp[i])

        # get patchsize around center; corner cases handled inside loop
        boxes[:, :2] = ((boxes[:, :2] + boxes[:, 2:] - spPatch) / 2)
        boxes = boxes.astype(np.int)
        boxes[:, 2:] = boxes[:, :2] + spPatch

        for j in range(boxes.shape[0]):
            # fix corner cases
            xmin, xmax = np.maximum(0, np.minimum(boxes[j, [0, 2]], w - 1))
            ymin, ymax = np.maximum(0, np.minimum(boxes[j, [1, 3]], h - 1))
            xmax = spPatch if xmin == 0 else xmax
            xmin = xmax - spPatch if xmax == w - 1 else xmin
            ymax = spPatch if ymin == 0 else ymax
            ymin = ymax - spPatch if ymax == h - 1 else ymin

            imPatch = im[i, ymin:ymax, xmin:xmax]
            hogF = hog(
                color.rgb2gray(imPatch), 
                orientations=hogBins,
                pixels_per_cell=(hogCellSize, hogCellSize),
                cells_per_block=(int(np.sqrt(hogCells)),int(np.sqrt(hogCells))),
                visualise=False)
            
            colHist = color_histogram_descriptor(imPatch, colBins)
            
            regions[count, :] = np.hstack((
                hogF, colHist, [boxes[j, 1] * 1. / h, boxes[j, 0] * 1. / w]))
            count += 1
        

        frameEnd[i] = count - 1
        if not redirect:
            sys.stdout.write('Descriptor computation: [% 5.1f%%]\r' %
                                (100.0 * float((i + 1) / n)))
            sys.stdout.flush()
    regions = regions[:count]
    eTime = time.time()
    print('Descriptor computation finished: %.2f s' % (eTime - sTime))

    return regions, frameEnd


def compute_nn(regions, frameEnd, F=15, L=4, redirect=False):
    """
    Compute transition matrix using nearest neighbors. This graph is 
    constructed using a kd-tree and searches for the 4 nearest 
    neighbors of every superpixel (using the created region 
    descriptor) over a spectral radius of 15 frames, including the 
    current frame. Thus, the 4 nearest neighbors of every region are 
    found in the 15 previous frames, the current frame, and the 
    following 15 frames. This results in 4(2*15 + 1) = 124 nearest 
    neighbors. See section 3.2, 'Detailed Description of the 
    Algorithm - Nearest Neighbor (NNs) Search' [2].

    INPUT:  regions: (k,d): k regions with d-dim feature
            frameEnd: (n,): indices of regions where frame ends: 
            0-indexed, included
            F: temporal radius: nn to be searched in (2F+1) frames 
            around curr frame
            L: nearest neighbors to be found per frame on an average
    
    OUTPUT: transM: (k,k)
    """

    sTime = time.time()
    M = L * (2 * F + 1)
    k, _ = regions.shape
    n = frameEnd.shape[0]
    transM = np.zeros((k, k), dtype=np.float)

    # Build 0-1 nn graph based on L2 distance using KDTree
    for i in range(n):
        # build KDTree with 2F+1 frames around frame i
        startF = max(0, i - F)
        startF = 1 + frameEnd[startF - 1] if startF > 0 else 0
        endF = frameEnd[min(n - 1, i + F)]
        tree = KDTree(regions[startF:1 + endF], leafsize=100)

        # find nn for regions in frame i
        currStartF = 1 + frameEnd[i - 1] if i > 0 else 0
        currEndF = frameEnd[i]
        distNN, nnInd = tree.query(regions[currStartF:1 + currEndF], M)
        nnInd += startF
        currInd = np.mgrid[currStartF:1 + currEndF, 0:M][0]
        transM[currInd, nnInd] = distNN
        if not redirect:
            sys.stdout.write('NearestNeighbor computation: [% 5.1f%%]\r' %
                                (100.0 * float((i + 1) / n)))
            sys.stdout.flush()

    eTime = time.time()
    print('NearestNeighbor computation finished: %.2f s' % (eTime - sTime))

    return transM


def normalize_nn(transM, sigma=1):
    """
    Want to votes to have weighting based on proximity. Do not 
    believe this is officially mentioned in either paper.
    Normalize transition matrix using gaussian weighing. 
    Input:
        transM: (k,k)
        sigma: var=sigma^2 of gaussian weight between elements
    Output: transM: (k,k)
    """
    # Make weights Gaussian and normalize
    k = transM.shape[0]
    transM[np.nonzero(transM)] = np.exp(
        -np.square(transM[np.nonzero(transM)]) / sigma**2)
    transM[np.arange(k), np.arange(k)] = 1.
    normalization = np.dot(transM, np.ones(k))
    transM = (1. / normalization).reshape((-1, 1)) * transM
    
    return transM


def isDominant(
        flow, flow_magnitude_thresh, flow_direction_thresh, direction_bins=10):
    """
    Look for frames where the dominant motion is close to zero. 
    First, check if the median of the optical flow magnitude is 
    below a certain threshold, denoted as flow_magnitude_thresh.
    Denote this as 'static' dominant motion'. 
    Then, check if the camera translation results in a dominant 
    direction. This is done by creating a histogram of the 
    optical flow orientations (directions) for each frame. 
    The bins are weighted according to the optical flow 
    magnitude. If the the bin with the most counts has a weight 
    above a certain threshold, denoted as flow_direction_thresh, 
    then it can be declared that the camera translation is some 
    dominant direction. Denote these frames as having 'dominant
    translation'. For a complete description of determining what the 
    dominant motion is, refer to section 4, Initializing the 
    Voting Scheme , under Motion Saliency Cues.[2].

    INPUTS: 
            flow_magnitude_thresh - Optical flow magnitude threshold. 
            In the original NLC paper, the authors use a value of 1.

            flow_direction_thresh - Optical flow direction threshold.
            In original NLC paper, the authors use a value of 0.75.

            direction_bins - Number of bins for the flow orientation 
            histogram.
    """

    # Initialize the motion type. This will indicate what the 
    # dominant motion is in a given frame. This will take label 
    # of either 'static' or 'translation'. Target will be used for 
    # determining the deviation from either 0 or the estimated 
    # translation direction.

    motion_type = ''
    dominant = False
    target = -1000 

    flow_magnitude = np.square(flow)
    flow_magnitude = np.sqrt(flow_magnitude[..., 0] + flow_magnitude[..., 1])
    flow_magnitude_median = np.median(flow_magnitude)
    
    # Test case for determining if the camera is static. 
    if flow_magnitude_median < flow_magnitude_thresh:
        dominant = True
        targetIm = flow_magnitude # targer IM? not sure
        target = 0. #again not sure
        motion_type = 'static'

    # Look for frames where the camera translation is in some 
    # dominant direction, given that dominant motion is not close 
    # to zero. 
    if not dominant:

        # Orientation in radians: (-pi, pi).
        flow_orientation = np.arctan2(flow[..., 1], flow[..., 0])

        # Compute global histogram of the optical flow orientations 
        # for each frame. Weight the bins by the optical flow 
        # magnitude. Normalize the results. 
        flow_orientation_histogram, bins = np.histogram(
            flow_orientation, bins=direction_bins,
            weights=flow_magnitude, range=(-np.pi, np.pi))
        
        flow_orientation_histogram /= (np.sum(flow_orientation) 
            + (np.sum(flow_orientation) == 0))
        
        # Test case for determining if the camera is translational.
        if np.max(flow_orientation_histogram) > flow_direction_thresh:
            dominant = True
            targetIm = flow_orientation
            target = (bins[np.argmax(flow_orientation_histogram)] 
                + bins[np.argmax(flow_orientation_histogram) + 1])
            target /= 2.
            motion_type = 'translate'

    # If dominant motion has been established, determine the 
    # amount each pixel within a patch deviates form the dominant 
    # motion. Use deviation as Var(X) = E[(X-mu)^2].
    if dominant:
        deviation = (targetIm - target)**2
        if moType == 'translate':
            
            # For orientation: theta = theta + 2pi. Thus,want min of:
            # theta1-theta2 = theta1-theta2-2pi = 2pi+theta1-theta2.
            deviation = np.minimum(deviation, (targetIm - target + 2.*np.pi)**2)
            deviation = np.minimum(deviation, (targetIm - target - 2.*np.pi)**2)
        
        saliency = convolve2d(deviation, np.ones((patchSz,patchSz))/patchSz**2,
            mode='same', boundary='symm')
        return dominant, moType, target, saliency

    return dominant, moType, target, -1000


def compute_saliency(
    imSeq, pyflow_parameters=False, flowSz=100, flowBdd=12.5, flowF=3, 
    flowWinSz=10, flowMagTh=1, flowDirTh=0.75, numDomFTh=0.5, flowDirBins=10, 
    patchSz=5, redirect=False, doNormalize=True, defaultToAppearance=True):
    """
    Initialize for FG/BG votes by Motion or Appearance Saliency. FG>0, BG=0.
    Input:
        imSeq: (n, h, w, c) where n > 1: 0-255: np.uint8: RGB
        flowSz: target size of image to be resized to for computing optical flow
        flowBdd: percentage of smaller side to be removed from bdry for saliency
        flowF: temporal radius to find optical flow
        flowWinSz: winSize in farneback (large -> get fast motion, but blurred)
        numDomFTh: # of dominant frames needed for motion Ssliency
        flowDirBins: # of bins in flow orientation histogram
        patchSz: patchSize for obtaining motion saliency score
    Output:
        salImSeq: (n, h, w) where n > 1: float. FG>0, BG=0. score in [0,1].
    """

    sTime = time.time()

    # pyflow Options:
    if pyflow_parameters == False:
        alpha = 0.012
        ratio = 0.75
        minWidth = 20
        nOuterFPIterations = 7
        nInnerFPIterations = 1
        nSORIterations = 30
    else:
        alpha = pyflow_parameters['alpha']
        ratio = pyflow_parameters['ratio']
        minWidth =pyflow_parameters['minWidth']
        nOuterFPIterations = pyflow_parameters['nOuterFPIterations']
        nInnerFPIterations = pyflow_parameters['nInnerFPIterations']
        nSORIterations = pyflow_parameters['nSORIterations']


    n, h, w, c = imSeq.shape
    im = np.zeros((n, flowSz, flowSz, c), np.uint8)

    # decrease size for optical flow computation
    for i in range(n):
        im[i] = imresize(imSeq[i], (flowSz, flowSz))

    # compute Motion Saliency per frame
    salImSeq = np.zeros((n, flowSz, flowSz))
    numDomFrames = 0
    for i in range(n):
        isFrameDominant = 0
        for j in range(-flowF, flowF + 1):
            if j == 0 or i + j < 0 or i + j >= n:
                continue
            # pyflow needs im: float in [0,1]
            u, v, _ = pyflow.coarse2fine_flow(
                im[i].astype(float) / 255., im[i + j].astype(float) / 255.,
                *pyflow_parameters, 0)
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)

            dominant, _, target, salIm = isDominant(
                flow, flowMagTh, flowDirTh, dirBins=flowDirBins)

            if dominant:
                salImSeq[i] += salIm
                isFrameDominant += 1

        if isFrameDominant > 0:
            salImSeq[i] /= isFrameDominant
            numDomFrames += isFrameDominant > 0
        if not redirect:
            sys.stdout.write('Motion Saliency computation: [% 5.1f%%]\r' %
                                (100.0 * float((i + 1) / n)))
            sys.stdout.flush()
    eTime = time.time()
    print('Motion Saliency computation finished: %.2f s' % (eTime - sTime))

    if numDomFrames < n * numDomFTh and defaultToAppearance:
        print('Motion Saliency not enough.. using appearance.')
        sTime = time.time()
        mr = MR.MR_saliency()
        for i in range(n):
            salImSeq[i] = mr.saliency(im[i])
            if not redirect:
                sys.stdout.write(
                    'Appearance Saliency computation: [% 5.1f%%]\r' %
                    (100.0 * float((i + 1) / n)))
                sys.stdout.flush()
        # Higher score means lower saliency. Correct it across full video !
        salImSeq -= np.max(salImSeq)
        eTime = time.time()
        print('Appearance Saliency computation finished: %.2f s' %
                (eTime - sTime))

    # resize back to image size, and exclude boundaries
    exclude = int(min(h, w) * flowBdd * 0.01)
    salImSeqOrig = np.zeros((n, h, w))
    for i in range(n):
        # bilinear interpolation to upsample back
        salImSeqOrig[i, exclude:-exclude, exclude:-exclude] = \
            ndimage.interpolation.zoom(
            salImSeq[i], (h * 1. / flowSz, w * 1. / flowSz), order=1)[
            exclude:-exclude, exclude:-exclude]

    # normalize full video, and NOT per frame
    if np.max(salImSeqOrig) > 0 and doNormalize:
        salImSeqOrig /= np.max(salImSeqOrig)

    return salImSeqOrig


def salScore2votes(salImSeq, sp):
    """
    Convert saliency score to votes
    Input:
        salImSeq: (n, h, w) where n > 1: float. FG>0, BG=0. score in [0,1].
        sp: (n,h,w): 0-indexed regions, #regions <= numsp
    Output:
        votes: (k,) where k < numsp*n
    """
    n, h, w = salImSeq.shape
    numsp = np.max(sp) + 1
    votes = np.zeros((numsp * n,), dtype=np.float)
    startInd = 0
    for i in range(n):
        sp1 = sp[i].reshape(-1)
        val1 = salImSeq[i].reshape(-1)
        sizeOut = np.max(sp1) + 1
        # assign average score of pixels to a superpixel
        sumScore = utils.my_accumarray(sp1, val1, sizeOut, 'plus')
        count = utils.my_accumarray(sp1, np.ones(sp1.shape), sizeOut, 'plus')
        votes[startInd:startInd + sizeOut] = sumScore / count
        startInd += sizeOut
    votes = votes[:startInd]

    return votes


def consensus_vote(votes, transM, frameEnd, iters):
    """
    Perform iterative consensus voting. The initial saliency map is 
    cast to the graph and an iterative voting procedure is conducted. 
    Each iteration consists of updating every region by a weighted 
    average of that region's 124 nearest neighbors. See section 3.1, 
    'The Algorithm' [2].
    """
    sTime = time.time()
    for t in range(iters):
        votes = np.dot(transM, votes)
        # normalize per frame
        for i in range(frameEnd.shape[0]):
            currStartF = 1 + frameEnd[i - 1] if i > 0 else 0
            currEndF = frameEnd[i]
            frameVotes = np.max(votes[currStartF:1 + currEndF])
            votes[currStartF:1 + currEndF] /= frameVotes + (frameVotes <= 0)
    eTime = time.time()
    print('Consensus voting finished: %.2f s' % (eTime - sTime))
    return votes


def votes2mask(votes, sp):
    """
    Project votes to images to obtain masks
    Input:
        votes: (k,) where k < numsp*n
        sp: (h,w) or (n,h,w): 0-indexed regions, #regions <= numsp
    Output:
        maskSeq: (h,w) or (n,h,w):float. FG>0, BG=0.
    """
    if sp.ndim < 3:
        sp = sp[None, ...]

    # operation is inverse of accumarray, i.e. indexing
    n, h, w = sp.shape
    maskSeq = np.zeros((n, h, w))
    startInd = 0
    for i in range(n):
        sp1 = sp[i].reshape(-1)
        sizeOut = np.max(sp1) + 1
        voteIm = votes[startInd:startInd + sizeOut]
        maskSeq[i] = voteIm[sp1].reshape(h, w)
        startInd += sizeOut

    if sp.ndim < 3:
        return maskSeq[0]
    return maskSeq


def remove_low_energy_blobs(maskSeq, binTh, relSize=0.6, relEnergy=None,
                                target=None):
    """
    Input:
        maskSeq: (n, h, w) where n > 1: float. FG>0, BG=0. Not thresholded.
        binTh: binary threshold for maskSeq for finding blobs: [0, max(maskSeq)]
        relSize: [0,1]: size of FG blobs to keep compared to largest one
                        Only used if relEnergy is None.
        relEnergy: Ideally it should be <= binTh. Kill blobs whose:
                    (total energy <= relEnergy * numPixlesInBlob)
                   If relEnergy is given, relSize is not used.
        target: value to which set the low energy blobs to.
                Default: binTh-epsilon. Must be less than binTh.
    Output:
        maskSeq: (n, h, w) where n > 1: float. FG>0, BG=0. Not thresholded. It
                 has same values as input, except the low energy blobs where its
                 value is target.
    """
    sTime = time.time()
    if target is None:
        target = binTh - 1e-5
    for i in range(maskSeq.shape[0]):
        mask = (maskSeq[i] > binTh).astype(np.uint8)
        if np.sum(mask) == 0:
            continue
        sp1, num = ndimage.label(mask)  # 0 in sp1 is same as 0 in mask i.e. BG
        count = utils.my_accumarray(sp1, np.ones(sp1.shape), num + 1, 'plus')
        if relEnergy is not None:
            sumScore = utils.my_accumarray(sp1, maskSeq[i], num + 1, 'plus')
            destroyFG = sumScore[1:] < relEnergy * count[1:]
        else:
            sizeLargestBlob = np.max(count[1:])
            destroyFG = count[1:] < relSize * sizeLargestBlob
        destroyFG = np.concatenate(([False], destroyFG))
        maskSeq[i][destroyFG[sp1]] = target
    eTime = time.time()
    print('Removing low energy blobs finished: %.2f s' % (eTime - sTime))
    return maskSeq


def nlc(imSeq, maxsp, iters, outdir, suffix='', pyflow_parameters=False, 
    clearBlobs=False, binTh=None, relEnergy=None, redirect=False, 
    doload=False, dosave=False):
    """
    Perform Non-local Consensus Voting (NLC) voting. 
    Input:
        imSeq: (n, h, w, c) where n > 1: 0-255: np.uint8: RGB
        maxsp: max # of superpixels per image
        iters: # of iterations of consensus voting
    Output:
        maskSeq: (n, h, w) where n > 1: float. FG>0, BG=0. Not thresholded.
    """

    if doload == True:
        dosave = False
    sys.setrecursionlimit(100000)

    if not doload:
        sp = region_extraction(imSeq, maxsp, redirect=redirect)
        regions, frameEnd = region_descriptor(imSeq, sp, redirect=redirect)
        transM = compute_nn(regions, frameEnd, F=15, L=2, redirect=redirect)
        transM = normalize_nn(transM, sigma=np.sqrt(0.1))
        salImSeq = compute_saliency(
            imSeq, pyflow_parameters, flowBdd=12.5, flowDirBins=20,redirect=redirect)

    suffix = outdir.split('/')[-1] if suffix == '' else suffix
    
    if doload:
        sp = np.load(outdir + '/sp_%s.npy' % suffix)
        regions = np.load(outdir + '/regions_%s.npy' % suffix)
        frameEnd = np.load(outdir + '/frameEnd_%s.npy' % suffix)
        transM = np.load(outdir + '/transM_%s.npy' % suffix)
        salImSeq = np.load(outdir + '/salImSeq_%s.npy' % suffix)
    
    if dosave:
        np.save(outdir + '/sp_%s.npy' % suffix, sp)
        np.save(outdir + '/regions_%s.npy' % suffix, regions)
        np.save(outdir + '/frameEnd_%s.npy' % suffix, frameEnd)
        np.save(outdir + '/transM_%s.npy' % suffix, transM)
        np.save(outdir + '/salImSeq_%s.npy' % suffix, salImSeq)
    

    # get initial votes from saliency salscores
    votes = salScore2votes(salImSeq, sp)
    assert votes.shape[0] == regions.shape[0], "Should be same, some bug !"

    # run consensus voting
    if clearBlobs and binTh is not None and relEnergy is not None:
        miniBatch = 5
        print('Intermediate blob removal is ON... %d times' % miniBatch)
        iterBatch = int(iters / miniBatch)
        for i in range(miniBatch):
            votes = consensus_vote(votes, transM, frameEnd, iterBatch)
            maskSeq = votes2mask(votes, sp)
            maskSeq = remove_low_energy_blobs(
                maskSeq, binTh=binTh, relEnergy=relEnergy, target=binTh / 4.)
            votes = salScore2votes(maskSeq, sp)
    else:
        votes = consensus_vote(votes, transM, frameEnd, iters)

    # project votes to images to obtain masks -- inverse of accumarray
    maskSeq = votes2mask(votes, sp)

    return maskSeq


def main():
    print('Please execute iss_main.py.')


if __name__ == "__main__":
    main()