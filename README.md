# Instrument Shaft Segmentation using Unsupervised Non-Local Consensus Voting (ISS-uNLC)

ISS-uNLC is an application of the methods explored in the 2017 paper, [Learning Features by Watching Objects Move](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). More specifically, it is an evaluation of the unsupervised bottom-up video motion segmentation algorithm, [uNLC]( https://github.com/pathak22/videoseg), explained in section 5.1 of the paper, on robotic surgery images and the ability to generate pseudo ground truth data for instrument shaft segmentation. As noted by the authors of Learning Features by Watching Objects Move, uNLC is in fact an implementation of Faktor and Irani’s NLC algorithm from their 2014 paper, [Video Segmentation by Non-Local Consensus Voting]( http://www.wisdom.weizmann.ac.il/~vision/NonLocalVideoSegmentation.html). 

This library includes what we consider a stripped down version of the available uNLC and choose to implement the aspects that are most relevant to our use. As a result, we impose a several modifications and offer a greater description of the algorithm that is being used. In our source code, we indicte the areas where these modifications take place. Furthermore, we attempt to provide detailed instructions on how to use the software.

We stress again that this is an implementation and adaptation of the available uNLC source code. Installation instructions are provided below for convenience and are specific to our use, however please view the authors’ original source code for complete installation instructions and demo. 

## Several Notes Regarding Installation and the Additional Libraries

As pyflow is a wrapper around [Ce Liu's C++ implementation of Coarse2Fine Optical Flow](http://people.csail.mit.edu/celiu/OpticalFlow/), the python wrapper utilizes the python package Cython. Cython consists of C-extensions for Python. When attempting to build the pyflow library, on a windows machine, we obtained the error of "error: Unable to find vcvarsall.bat". After attempting to troubleshoot, it appears that this is a common error and the culprit is with Visual Studio. At the time of study, we were unable to find a solution for this error that worked for us and decided instead to work on an alternative system. We decided to work on an Ubuntu 16.04 LTS system.  

## Required Libraries and Additional Dependencies

  ```Shell
  cv2
  Cython
  numpy
  PIL
  scipy
  skimage

  distutils
  glob
  os
  sys
  ```

## Installation Instructions
The installation instructions mimic those of Pathak’s videoseg, however we neglect the installation of the Dense CRF code, the Kernel Temporal Segmentation code, and DeepMatching.

1. Download and install uNLC.
  ```Shell
  cd ISS_uNLC/lib/
  git clone https://github.com/pathak22/videoseg.git
  ```

2. Download and install pyflow.
  ```Shell
  cd ISS_uNLC/lib/
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  ```

3. Download and install mr_saliency.
  ```Shell
  cd ISS_uNLC/lib/
  git clone https://github.com/ruanxiang/mr_saliency.git
  cp __init__.py mr_saliency/
  ```
  
In order to make mr_saliency/MR.py compatible with current library versions and python3, several changes need to be made. The first is a change from ‘<>’ to ‘!=’ for the ‘not equal’  conditional operator. Python3 no longer supports ‘<>’. The second change is to remove the importing of the lena image from the skimage.data module. The lena image has been removed from skimage due to copyrights. A simple fix, without going through the rest of the script is to import a different image from the skimage.data model as lena. For example, ‘from skimage.data import astronaut as lena’. To make this process simpler, a modified MR.py script is included with in this repository and can be swapped for the one in the mr_saliency library. These are the only required changes for mr_saliency.

The library should be ready to use. In order to check this, run the iss_checkpaths.py script. 
Python iss_checkpaths.py

If there are any errors. The paths have not been initialized correctly. If passed, the libray is ready to use for instrument shaft segmentation. 

There are three ways to import data to perform instrument segmentation. These include:

  
  
  
## Observations and Comments on Use
One observation which we came across was the difference in computation time between OpenCV and skikit-image for the SLIC superpixel construction. While the original authors note that it takes 2.5 seconds per 720x1280 image where there is a max of 200 labelled segments. 
