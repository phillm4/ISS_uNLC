## Instrument Shaft Segmentation using Unsupervised Non-Local Consensus Voting

Application of the methods explored in the 2017 paper, [Learning Features by Watching Objects Move](https://people.eecs.berkeley.edu/~pathak/unsupervised_video/). More specifically, it is an evaluation of the unsupervised bottom-up video motion segmentation algorithm, [uNLC]( https://github.com/pathak22/videoseg), explained in section 5.1 of the paper, on robotic surgery images and the ability to generate pseudo ground truth data for instrument shaft segmentation. As noted by the authors of Learning Features by Watching Objects Move, uNLC is in fact an implementation of Faktor and Irani’s NLC algorithm from their 2014 paper, [Video Segmentation by Non-Local Consensus Voting]( http://www.wisdom.weizmann.ac.il/~vision/NonLocalVideoSegmentation.html). 

This library includes what we consider a stripped down version of the available uNLC and choose to implement the aspects that are most relevant to our use. As a result, we impose a several modifications and offer a greater description of the algorithm that is being used. In our source code, we indicte the areas where these modifications take place. Furthermore, we attempt to provide detailed instructions on how to use the software.

We stress again that this is an implementation and adaptation of the available uNLC source code. Installation instructions are provided below for convenience and are specific to our use, however please view the authors’ original source code for complete installation instructions and demo. 

# Installation Instructions

1. Download and install uNLC.
  ```Shell
  git clone https://github.com/pathak22/videoseg
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```

2. Download and install pyflow.
  ```Shell
  git clone https://github.com/pathak22/pyflow.git
  cd pyflow/
  python setup.py build_ext -i
  python demo.py    # -viz option to visualize output
  ```
