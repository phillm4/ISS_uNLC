# Instrument Shaft Segmentation using Unsupervised Non-Local Consensus Voting

Application of the methods explored in the 2017 paper, [Learning Features by Watching Objects Move](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). More specifically, it is an evaluation of the unsupervised bottom-up video motion segmentation algorithm, [uNLC]( https://github.com/pathak22/videoseg), explained in section 5.1 of the paper, on robotic surgery images and the ability to generate pseudo ground truth data for instrument shaft segmentation. As noted by the authors of Learning Features by Watching Objects Move, uNLC is in fact an implementation of Faktor and Irani’s NLC algorithm from their 2014 paper, [Video Segmentation by Non-Local Consensus Voting]( http://www.wisdom.weizmann.ac.il/~vision/NonLocalVideoSegmentation.html). 

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
