## Instrument Shaft Segmentation using Unsupervised Non-local Consensus Voting

Instrument shaft segmentation for surgical robotic images was performed using the methods described in Pathak et al.’s 2017 Conference on Computer Vision and Pattern Recognition (CVPR) paper, [''Learning Features by Watching Objects Move''](http://cs.berkeley.edu/~pathak/unsupervised_video/). In particular, pseudo ground truth data for instrument segmentation was generated using the [unsupervised motion-based segmentation algorithm](https://github.com/pathak22/videoseg) presented in section 5.1 of Pathak et al. Convolutional neural networks (CNNs) trained from pseudo ground truth data have been shown to outperform the leading unsupervised methods for object detection. It is desired that the obtained pseudo ground truth data for instrument segmentation will be used to train Fully Convolutional Networks (FCNs) to perform semantic segmentation. 

The unsupervised motion-based segmentation algorithm is largely inspired by Faktor and Irani’s 2014 British Machine Vision Conference (BMVC) paper, [''Video Segmentation by Non-Local Consensus Voting''](http://www.wisdom.weizmann.ac.il/~vision/NonLocalVideoSegmentation.html). Pathak et al.’s algorithm, denoted as uNLC, differs from Faktor and Irani’s algorithm, denoted as NLC, as uNLC substitutes a trained edge detector for an unsupervised superpixel generator. 

Using the methods described by Pathak et al. and Faktor & Irani, pseudo ground truth data for instrument shaft segmentations was obtained and evaluated. The installation process and instructions on using the software is described below.

### Disclaimer
The majority of the software used for this project is from Pathak's [videoseg](https://github.com/pathak22/videoseg) library. While a handful of modifications and changes were introduced, the software is ultimately under the ownership of Pathak et al., the authors of [''Learning Features by Watching Objects Move''](http://cs.berkeley.edu/~pathak/unsupervised_video/).
```
@inproceedings{pathakCVPR17learning,
    Author = {Pathak, Deepak and Girshick, Ross and Doll\'{a}r,
              Piotr and Darrell, Trevor and Hariharan, Bharath},
    Title = {Learning Features by Watching Objects Move},
    Booktitle = {Computer Vision and Pattern Recognition ({CVPR})},
    Year = {2017}
}
```

### Description of Contents
| File / Folder         | Description                                                                   |
| --------------------- |:------------------------------------------------------------------------------| 
| additional\_tools     | Contains extra scripts that may be beneficial                                 | 
| lib                   | Location where all external code will be stored                               |  
| modified\_scripts     | Contains several scripts that will be placed into external code libraries     | 
| \_init\_nlc\_path.py  | Script which initializes library paths for *iss\_main.py*                     |
| iss\_main.py          | Program to initiate uNLC algorithm                                            |
| test\_batch.zip       | Zipped folder containing test images for the program demo                     |

It is intend in the future to remove the bulk of the dependencies on scikit-image and PIL, and instead use OpenCV exclusively. This process has already been initiated.

### Required Libraries and Additional Dependencies
In order to run the python scripts for uNLC, several additional libraries are required. One of the required libraries is pyflow. Pyflow is a wrapper around [Ce Liu's C++ implementation of Coarse2Fine Optical Flow](http://people.csail.mit.edu/celiu/OpticalFlow/), and utilizes the python package Cython. Cython consists of C-extensions for Python. When attempting to build the pyflow library on a windows machine, "error: Unable to find vcvarsall.bat" was encountered. This appears to be a common error due to Visual Studio. As a result, all work was completed on an Ubuntu 16.04 LTS system. Additional libraries include the following:

```
    OpenCV3 - cv2
    Cython
    numpy
    Python Imaging Library - PIL
    scipy
    scikit-image - skimage
```

The majority of the libraries can be installed via pip or conda with the exception of OpenCV. While the uNLC algorithm does work without the FFmpeg dependencies, it is required if video files are to be read. If it is not desired to work with videos, the pip or conda installation of OpenCV should work. 

### Installation Instructions
The installation instructions mimic those of Pathak’s videoseg, however the installations of Dense CRF, Kernel Temporal Segmentation, and DeepMatching are all neglected. These packages need to be installed if it is desired to run Pathak’s *full\_pipe.py* script. In addition, several modifications are required in order for functions to work properly. A handful of scripts have been included as an attempt to mitigate these issues.

##### Notation.
Throughout the installation procedure, the bulk of the user path will be omitted. For example, `/home/user/path/$    ` will be referred to as `$    `, and commands like, `$ cd /home/user/path/ISS_uNLC/lib/ ` will be indicated by `$ cd ISS_uNLC/lib/ `.

##### 1. Download and install this repository.
```
$ cd [Path where this will be installed]
$ git clone https://github.com/phillm4/ISS_uNLC.git
```
##### 2. Download and install NLC. 
As this will be used as a python library, a *\_\_init\_\_.py* file needs to be included.
```
$ cd ISS_uNLC/lib/
$ git clone https://github.com/pathak22/videoseg.git
$ cp __init__.py videoseg/
```

##### 3.Download and install pyflow.
Note that the installation path has changed. Furthermore, it is required to build pyflow as it is a C++ wrapper. 
```
$ cd ISS_uNLC/lib/videoseg/lib/
$ git clone https://github.com/pathak22/pyflow.git
$ cd pyflow/
$ python setup.py build_ext -i
```

##### 4. Download and install Visual Saliency. 
Similar to NLC, a *\_\_init\_\_.py* file needs to be included. Note the installation path.
```
$ cd ISS_uNLC/lib/videoseg/lib/
$ git clone https://github.com/ruanxiang/mr_saliency.git
$ cp __init__.py mr_saliency/
```

After installing the above dependencies, several modifications are required for everything to work properly.

##### 5. Fix MR.py issues
*mr\_saliency/MR.py* handles the visual saliency calculation in uNLC. However, the script is not compatible with Python 3 nor the current version of scikit-image. In order to fix these issues, `<>` needs to be changed to `!=`. Python 3 no longer supports `<>` as a comparison operator. The second change is to remove the importing of `lena` from the skimage.data module. lena has been removed from scikit-image due to copyright issues. A simple fix is to import a different image from the skimage.data module and name that as lena. To make this process simpler, a modified *MR.py* script is included in this repository and can be placed in the *mr\_saliency* library. This procedure is shown below.

```
$ cd ISS_uNLC/
$ cp modified_scripts/MR_mod.py lib/videoseg/lib/mr_saliency/
```

##### 6. Add nlc\_mod.py
The last step in the installation process is to move the included *nlc\_mod.py* script into *videoseg/lib*. *nlc\_mod.py* is a modified version of *videoseq/src/NLC.py* which allows for the tuning of the pyflow parameters. It is intended to update *nlc\_mod.py* in the future as to remove this step in the installation process and to remove additional dependencies.
```
$ cd ISS_uNLC/
$ cp modified_scripts/nlc_mod.py lib/videoseg/src/
```

uNLC should be ready to use.

### Usage Instructions
All operations are handled by *iss\_main.py*. This is the main program to run the uNLC algorithm and is a wrapper around Pathak's nlc algorithm. It is inspired by the NLC library's *run\_full.py* and *nlc.py*. The function can be executed from the command line and accepts a variety of arguments. In general, *iss\_main.py* requires an input corresponding to the path of either an image directory, a directory containing subfolders with images, or a video. Additional arguments include the ability to adjust the output directory and a frame gap. Use `-h` to view the arguments.

```
$ cd ISS_uNLC/
$ python iss_main.py -h

usage: iss_main.py [-h] [-indir INDIR] [-outdir OUTDIR] [-batch BATCH]
                   [-vid VID] [-fgap FGAP]

Wrapper for Pathak's Non-Local Consensus Voting algorithm for Instrument Shaft
Segmentation.

optional arguments:
  -h, --help      show this help message and exit
  -indir INDIR    Path to the input directory that contains the images to
                  test.
  -outdir OUTDIR  Path to the output directory to save results. If no
                  directory is specified, 'results' directory will be created
                  at the same level as the input directory.
  -batch BATCH    Batch input. Path to a directory that contains sub-folders
                  containing images to test.
  -vid VID        Path to the video file to be converted. Include the video
                  itself. If inputting a video, the video willbe split into
                  subfolders to conserve memory when running uNLC.
  -fgap FGAP      Frame gap between images in a sequence. Default 0.
```

Note that there are three different methods for inputting data, and two additional arguments which correspond to an output directory and a frame gap. The frame gap is important when specifying the frames that uNLC is to compute across in a given image sequence. If a frame gap too small is chosen, memory errors may occur and the computation may take a long time (up to an hour). When using uNLC, it is important to be aware of the length of image sequences and the frame gap as to prevent memory issues. If a frame gap too large is chosen, the obtained results will be useless. Furthermore, uNLC works best on image sequences where there is significant motion by the foreground objects. If the objects are barely moving, or move relatively slow, it may be beneficial to increase the frame gap. The output directory is where the segmented images will be saved. If this is not specified, a results directory will be created at the same level of any given input. Two examples are shown below.

#### Example (1.) 
Perform segmentation using the `-batch` input. In the context of *iss\_main.py*, this option is to be selected if there is a folder that contains several subfolders, each of which that contains images. This is illustrated as the following.
```
- batch_folder/
-- image_directory_00/
--- img_00_00.jpg
--- img_00_01.jpg
--- ...
-- image_directory_01/
--- img_01_00.jpg
--- img_01_01.jpg
--- ...
-- ...
```
In order to then perform segmentation on this batch, the commands and potential output are shown below. For this example, the included *test\_batch/* will be used as the input batch folder, the output directory will be generated automatically, and a frame gap of 3 frames will be used. This process will take several minutes to complete.
```
$ cd ISS_uNLC/
$ unzip test_batch
$ rm test_batch.zip
$ python iss_main.py -batch test_batch -fgap 3

Batch:  0
Input Directory:  /home/.../ISS_uNLC/test_batch/00
Output Directory:  /home/.../ISS_uNLC/results
Loading images: [ ##.# %]
Total Sequence Shape:  (##, ###, ###, #)
Memory Usage for Sequence: ##.## MB.

*****Performing uNLC*****

Superpixel computation: [ ##.# %]
Superpixel computation finished: #.## s

Descriptor computation: [ ##.# %]
Descriptor computation finished: #.## s

NearestNeighbor computation: [ ##.# %]
NearestNeighbor computation finished: ##.## s

Constructing pyramid...done!
Pyramid level 4
Pyramid level 3
Pyramid level 2
Pyramid level 1
Pyramid level 0
Constructing pyramid...done!
Pyramid level 4
Pyramid level 3
Pyramid level 2
Pyramid level 1
Pyramid level 0
Constructing pyramid...done!

Motion Saliency computation finished: #.## s
Consensus voting finished: #.## s

*****uNLC complete. Save results.*****

Removing low energy blobs finished: #.## s

Batch:  1
Input Directory:  /home/.../vidpath/src_images/00
Output Directory:  /home/.../vidpath/results
Loading images: [ ##.# %]
Total Sequence Shape:  (##, ###, ###, #)
Memory Usage for Sequence: ##.## MB.
...
```
Once the process is complete, the segmentation results will be located in the created *results/* folder.

#### Example (2.) 
Segment a **short** video that is located at *home/.../vidpath/video.avi* (not included in this repository). No output directory will be specified and a frame gap of 3 will be used. Segmenting directly from a video is not recommended at this time and it is important that the video is short.
```
$ cd ISS_uNLC/
$ python iss_main.py -vid home/.../vidpath/video.avi -fgap 3

Video to Image Conversion: [ ##.# %]

Batch:  0
Input Directory:  /home/.../vidpath/src_images/00
Output Directory:  /home/.../vidpath/results
...
```
The output should mimic that of Example 1. 

Once the process is complete (it may take several minutes depending on the length of the image sequence), two new folders *src\_images/* and *results/* should have been created where the video is located. One contains the frames from the video (*src\_images/*), the other contains the segmentation results (*results/*). The size of the batches created by the video to image conversion can be edited and is denoted as [*images\_per\_batch*](https://github.com/phillm4/ISS_uNLC/blob/5b7681b837a87745ed8e648dbedb0fa881e181db/iss_main.py#L213).

### Tuning uNLC

If it is desired to adjust any of the parameters for the uNLC algorithm besides the frame gap, these can be modified within *iss\_main.py* under the [iss\_uNLC() function](https://github.com/phillm4/ISS_uNLC/blob/edc62b3c5af0547940d9d7184769de3fcc252631/iss_main.py#L322). 

```python
    ## # Parameters for uNLC and pyflow.
    # memory_limit - Memory limit of image sequence for numpy array. 
    # resize_fraction -  Fraction that image will be resized by.
    # max_superpixels - Total number of superpixel regions.
    # vote_iterations - Number of times to perform consensus voting. 
    # segmentation_energy_threshold - Threshold for finding foreground objects.
    # relative_energy - Remove objects where: 
    #   (total energy <= relative_energy * foreground_pixel_size) 

    memory_limit = 100
    resize_fraction = 0.5
    max_superpixels = 1000
    vote_iterations = 100
    segmentation_energy_threshold = 0.3
    relative_energy = segmentation_energy_threshold - 0.1 
    clearVoteBlobs = False  
    clearFinalBlobs = True
    pyflow_parameters = dict(
        alpha = 0.012,
        ratio = 0.75,
        minWidth = 20,
        nOuterFPIterations = 7,
        nInnerFPIterations = 1,
        nSORIterations = 30)
```

### Next Steps
This concludes the installation and use instructions for uNLC. Several changes to the algorithm are planned for the future. It is intended to use the generated results to train a Fully Convolutional Network to perform instrument shaft segmentation. Additional scripts including a pytorch example and a slic superpixel demo can be found in the *additional\_tools* directory. 
