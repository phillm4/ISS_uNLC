"""
This file implements following paper:
Video Segmentation by Non-Local Consensus Voting

**********************************************************************
Disclaimer: This script is a modification of Pathak's nlc.py script.
This implementation is intended to provide further clarification 
and provide information referring back to the journal articles where 
the algorithm originated from.

References: 
[1] - Pathak et al., Learning Features by Watching Objects Move, 2017.
[2] - Faktor and Irani, Video Segmentation by Non-Local Consensus 
Voting, 2014
**********************************************************************
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import os
import sys
from PIL import Image
import numpy as np
from skimage.segmentation import slic
from skimage.feature import hog
from skimage import color
from scipy.spatial import KDTree
from scipy.misc import imresize
from scipy import ndimage
# from cv2 import calcOpticalFlowFarneback, OPTFLOW_FARNEBACK_GAUSSIAN
from scipy.signal import convolve2d
import time
import utils
import _init_paths  # noqa
from mr_saliency import MR_mod as MR
import pyflow


def superpixels(im, maxsp=200, vis=False, redirect=False):
    """
    Region extraction process. uNLC obtains regions through SLIC 
    superpixel segmentation. This stage in the algorithm is where 
    uNLC and NLC differ from one another. Where NLC adopts a trained 
    edge detector, uNLC instead performs SLIC. The original region 
    extraction process can be found in section 3.2, 'Detailed 
    Description of the Algorithm' under 'Region Extraction' [2]. 
    For uNLC, this is described in section 5.1, 'Unsupervised Motion 
    Segmentation' [1].

    Get Slic Superpixels
    Input: im: (h,w,c) or (n,h,w,c): 0-255: np.uint8: RGB
    Output: sp: (h,w) or (n,h,w): 0-indexed regions, #regions <= maxsp
    """
    sTime = time.time()
    if im.ndim < 4:
        im = im[None, ...]
    sp = np.zeros(im.shape[:3], dtype=np.int)
    for i in range(im.shape[0]):
        # slic needs im: float in [0,1]
        sp[i] = slic(im[i].astype(np.float) / 255., n_segments=maxsp, sigma=5)
        if not redirect:
            sys.stdout.write('Superpixel computation: [% 5.1f%%]\r' %
                                (100.0 * float((i + 1) / im.shape[0])))
            sys.stdout.flush()
    eTime = time.time()
    print('Superpixel computation finished: %.2f s' % (eTime - sTime))

    if vis and False:
        # TODO: set directory to save
        from skimage.segmentation import mark_boundaries
        for i in range(im.shape[0]):
            Image.fromarray((mark_boundaries(im[i], sp[i]))).save('.jpg')

    if im.ndim < 4:
        return sp[0]
    return sp


def get_region_boxes(sp):
    """
    Get bounding boxes for each superpixel image
    Input: sp: (h,w): 0-indexed regions, #regions <= numsp
    Output: boxes: (numsp, 4) : (xmin, ymin, xmax, ymax)
    """
    x = np.arange(0, sp.shape[1])
    y = np.arange(0, sp.shape[0])
    xv, yv = np.meshgrid(x, y)
    sizeOut = np.max(sp) + 1
    sp1 = sp.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    spxmin = utils.my_accumarray(sp1, xv, sizeOut, 'min')
    spymin = utils.my_accumarray(sp1, yv, sizeOut, 'min')
    spxmax = utils.my_accumarray(sp1, xv, sizeOut, 'max')
    spymax = utils.my_accumarray(sp1, yv, sizeOut, 'max')

    boxes = np.hstack((spxmin.reshape(-1, 1), spymin.reshape(-1, 1),
                        spxmax.reshape(-1, 1), spymax.reshape(-1, 1)))
    return boxes


def color_hist(im, colBins):
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

    Get color histogram descriptors for RGB and LAB space.
    Input: im: (h,w,c): 0-255: np.uint8: RGB
    Output: descriptor: (colBins*6,)
    """
    assert im.ndim == 3 and im.shape[2] == 3, "image should be rgb"
    arr = np.concatenate((im, color.rgb2lab(im)), axis=2).reshape((-1, 6))
    desc = np.zeros((colBins * 6,), dtype=np.float)
    for i in range(3):
        desc[i * colBins:(i + 1) * colBins], _ = np.histogram(
            arr[:, i], bins=colBins, range=(0, 255))
        desc[i * colBins:(i + 1) * colBins] /= np.sum(
            desc[i * colBins:(i + 1) * colBins]) + (
            np.sum(desc[i * colBins:(i + 1) * colBins]) < 1e-4)
    i += 1
    desc[i * colBins:(i + 1) * colBins], _ = np.histogram(
        arr[:, i], bins=colBins, range=(0, 100))
    desc[i * colBins:(i + 1) * colBins] /= np.sum(
        desc[i * colBins:(i + 1) * colBins]) + (
        np.sum(desc[i * colBins:(i + 1) * colBins]) < 1e-4)
    for i in range(4, 6):
        desc[i * colBins:(i + 1) * colBins], _ = np.histogram(
            arr[:, i], bins=colBins, range=(-128, 127))
        desc[i * colBins:(i + 1) * colBins] /= np.sum(
            desc[i * colBins:(i + 1) * colBins]) + (
            np.sum(desc[i * colBins:(i + 1) * colBins]) < 1e-4)
    return desc


def compute_descriptor(im, sp, spPatch=15, colBins=20, hogCells=9,
                        hogBins=6, redirect=False):
    """
    Region descriptors are computed for each superpixel. 
    The descriptor is made up of a concatenation of four 
    measurements: RGB color histogram and LAB color histogram (6 
    channels, 20 bins each), histogram of oriented gradients 
    (9 cells, 6 orientation bins, computed over a 15x15 patch around 
    the superpixel) and the relative spatial coordinates of the 
    superpixel. See section 3.2, 'Detailed Description of the 
    Algorithm - Region Descriptor' [2].
    Compute region descriptors for NLC
    Input:
        im: (h,w,c) or (n,h,w,c): 0-255: np.uint8: RGB
        sp: (h,w) or (n,h,w): 0-indexed regions, #regions <= numsp
        spPatch: patchsize around superpixel for feature computation
    Output:
        regions: (k,d) where k < numsp*n
        frameEnd: (n,): indices of regions where frame ends: 0-indexed, included
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
                color.rgb2gray(imPatch), orientations=hogBins,
                pixels_per_cell=(hogCellSize, hogCellSize),
                cells_per_block=(int(np.sqrt(hogCells)),
                                    int(np.sqrt(hogCells))),
                visualise=False)
            colHist = color_hist(imPatch, colBins)
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
    Input:
        regions: (k,d): k regions with d-dim feature
        frameEnd: (n,): indices of regions where frame ends: 0-indexed, included
        F: temporal radius: nn to be searched in (2F+1) frames around curr frame
        L: nearest neighbors to be found per frame on an average
    Output: transM: (k,k)
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
    Normalize transition matrix using gaussian weighing
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
    # This is inefficient, bottom line is better ..
    # transM = np.dot(np.diag(1. / normalization), transM)
    transM = (1. / normalization).reshape((-1, 1)) * transM
    return transM


def compute_saliency(imSeq, pyflow_parameters=False, flowSz=100, flowBdd=12.5, flowF=3, flowWinSz=10,
                        flowMagTh=1, flowDirTh=0.75, numDomFTh=0.5,
                        flowDirBins=10, patchSz=5, redirect=False,
                        doNormalize=True, defaultToAppearance=True):
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

    def isDominant(flow, flowMagTh, flowDirTh, dirBins=10):
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
        """
        mag = np.square(flow)
        mag = np.sqrt(mag[..., 0] + mag[..., 1])
        med = np.median(mag)
        dominant = False
        target = -1000
        moType = ''
        if med < flowMagTh:
            dominant = True
            targetIm = mag
            target = 0.
            moType = 'static'

        if not dominant:
            # orientation in radians: (-pi, pi): disambiguates sign of arctan
            orien = np.arctan2(flow[..., 1], flow[..., 0])
            # use ranges, number of bins and normalization to compute histogram
            dirHist, bins = np.histogram(orien, bins=dirBins, weights=mag,
                                            range=(-np.pi, np.pi))
            dirHist /= np.sum(dirHist) + (np.sum(dirHist) == 0)
            if np.max(dirHist) > flowDirTh:
                dominant = True
                targetIm = orien
                target = bins[np.argmax(dirHist)] + bins[np.argmax(dirHist) + 1]
                target /= 2.
                moType = 'translate'

        if dominant:
            # E[(x-mu)^2]
            deviation = (targetIm - target)**2
            if moType == 'translate':
                # for orientation: theta = theta + 2pi. Thus, we want min of:
                # (theta1-theta2) = (theta1-theta2-2pi) = (2pi+theta1-theta2)
                deviation = np.minimum(
                    deviation, (targetIm - target + 2. * np.pi)**2)
                deviation = np.minimum(
                    deviation, (targetIm - target - 2. * np.pi)**2)
            saliency = convolve2d(
                deviation, np.ones((patchSz, patchSz)) / patchSz**2,
                mode='same', boundary='symm')
            return dominant, moType, target, saliency

        return dominant, moType, target, -1000

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
            # flow = calcOpticalFlowFarneback(
            #     color.rgb2gray(im[i]), color.rgb2gray(im[i + j]), 0.5, 4,
            #     flowWinSz, 10, 5, 1.1, OPTFLOW_FARNEBACK_GAUSSIAN)
            # pyflow needs im: float in [0,1]
            u, v, _ = pyflow.coarse2fine_flow(
                im[i].astype(float) / 255., im[i + j].astype(float) / 255.,
                alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                nSORIterations, 0)
            flow = np.concatenate((u[..., None], v[..., None]), axis=2)

            dominant, _, target, salIm = isDominant(
                flow, flowMagTh, flowDirTh, dirBins=flowDirBins)

            if False:
                odir = '/home/dpathak/local/data/trash/my_nlc/nlc_out/'
                np.save(odir + '/np/outFlow_%d_%d.npy' % (i, i + j), flow)
                import cv2
                hsv = np.zeros((100, 100, 3), dtype=np.uint8)
                hsv[:, :, 0] = 255
                hsv[:, :, 1] = 255
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(odir + '/im/outFlow_%d_%d.png' % (i, i + j), rgb)

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
            clearBlobs=False, binTh=None, relEnergy=None,
            redirect=False, doload=False, dosave=False):
    """
    Perform Non-local Consensus voting moving object segmentation (NLC)
    Input:
        imSeq: (n, h, w, c) where n > 1: 0-255: np.uint8: RGB
        maxsp: max # of superpixels per image
        iters: # of iterations of consensus voting
    Output:
        maskSeq: (n, h, w) where n > 1: float. FG>0, BG=0. Not thresholded.
    """
    if dosave is None:
        dosave = not doload
    import sys
    sys.setrecursionlimit(100000)

    if not doload:
        # compute Superpixels -- 2.5s per 720x1280 image for any maxsp
        sp = superpixels(imSeq, maxsp, redirect=redirect)

        # compute region descriptors
        regions, frameEnd = compute_descriptor(imSeq, sp, redirect=redirect)

        # compute nearest neighbors
        transM = compute_nn(regions, frameEnd, F=15, L=2, redirect=redirect)

        # get initial saliency score: either Motion or Appearance Saliency
        salImSeq = compute_saliency(imSeq,pyflow_parameters=pyflow_parameters, flowBdd=12.5, flowDirBins=20,
                                        redirect=redirect)

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

    # create transition matrix
    transM = normalize_nn(transM, sigma=np.sqrt(0.1))

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

if __name__ == "__main__":
    print('Please execute iss_main.py.')

