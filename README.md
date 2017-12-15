## Disclaimer
The majority of the software used for this project is from [insert link]. A handful of modifications and changes were introduced, but the software is ultimately under the ownership of Pathak et al. 

## Description of Contents
It is intend in the future to remove the bulk of the dependencies on scikit-image and PIL, and instead use OpenCV exclusively. This process has already been initiated.

## Required Libraries and Additional Dependencies
In order to run the python scripts for uNLC, several additional libraries are required. One of the required libraries if pyflow. Pyflow is a wrapper around [Ce Liu's C++ implementation of Coarse2Fine Optical Flow] http://people.csail.mit.edu/celiu/OpticalFlow/), and the python wrapper utilizes the python package Cython. Cython consists of C-extensions for Python. When attempting to build the pyflow library on a windows machine, the error, "error: Unable to find vcvarsall.bat" was encountered. This appears to be a common error due to Visual Studio. As a result, all work was completed on an Ubuntu 16.04 LTS system.  
```Shell
  cv2
  Cython
  numpy
  PIL
  scipy
  scikit-image
```

The Majority of the libraries can be installed via pip or conda with the exception of OpenCV. While the uNLC algorithm does work without the FFmpeg dependencies, it is required if video files are to be read. 

## Installation Instructions
\noindent The installation instructions mimic those of Pathak’s videoseg, however the installations of Dense CRF, Kernel Temporal Segmentation, and DeepMatching are all neglected. These packages need to be installed if it is desired to run Pathak’s full\_pipe.py script. In addition, several modifications are required in order for functions to work properly. A handful of scripts have been included as an attempt to mitigate any issues.

### 1. Download and install this repository.
```Shell
$ cd [Path where this will be installed]
$ git clone https://github.com/pathak22/videoseg.git
```
### 2. Download and install NLC. 
As this will be used as a python library and consists of a variety of variety of python scripts, a \texttt{\_\_init\_\_.py} file needs to be included.
```Shell
$ cd ISS_uNLC/lib/
$ git clone https://github.com/pathak22/videoseg.git
$ cp __init__.py videoseg/
```
### 3.Download and install pyflow.
Note that the installation path has changed. Furthermore, it is required to build pyflow as it is a C++ wrapper. 
```Shell
$ cd ISS_uNLC/lib/videoseg/lib/
$ git clone https://github.com/pathak22/pyflow.git
$ cd pyflow/
$ python setup.py build_ext -i
```
### 4. Download and install Visual Saliency. 
Similar to NLC, a \texttt{\_\_init\_\_.py} file needs to be included.
```Shell
$ cd ISS_uNLC/lib/videoseg/lib/
$ git clone https://github.com/ruanxiang/mr_saliency.git
$ cp __init__.py mr_saliency/
```

After installing the above dependencies, several modifications are required in order for everything to work properly.

### 5. Fix MR.py issues
\texttt{mr\_saliency/MR.py} handles the visual saliency calculation in uNLC. However, the script is not compatible with Python 3 and the current version of scikit-image. In order to fix these issues, `$<>$' needs to be changed to `$!=$' for the `not equal' comparison operator as Python 3 no longer supports `$<>$'. The second change is to remove the importing of \texttt{lena} from the \texttt{skimage.data module}. \texttt{lena} has been removed from scikit-image due to copyright issues. A simple fix, without going through the rest of the script, is to import a different image from the \texttt{skimage.data model} and name that as \texttt{lena}. To make this process simpler, a modified \texttt{MR.py} script is included in this repository and can be swapped for the one in the \texttt{mr\_saliency} library. These are the only required changes for \texttt{mr\_saliency}. The swapping procedure is shown below.

```Shell
$ cd ISS_uNLC/
$ rm lib/mr_saliency/MR.py
$ mv MR.py lib/mr_saliency/
```

### 6. Add iss_uNLC.py
The last step in the installation process is to move the included \texttt{iss\_uNLC.py} script into the videoseg/lib. \texttt{iss\_uNLC.py} is a modified version of \texttt{videoseq/src/NLC.py} which allows for the tuning of the pyflow parameters. It is intended to update texttt{iss\_uNLC.py} in the future as to remove this step in the installation process.
```Shell
$ cd ISS_uNLC/lib/videoseg/lib/
$ cp iss\_uNLC.py lib/mr_saliency/
```

uNLC should be ready to use.

## Usage Instructions
All operations are handled by \texttt{iss\_main.py}. As previously indicated, this is the main program to run the uNLC algorithm and is a wrapper around Pathak's nlc algorithm. It is inspired by the NLC library's run\_full.py and nlc.py. The function can be executed from the command line and excepts a variety of arguments. In general, \texttt{iss\_main.py} requires an input corresponding to the path of either an image directory, a directory containing subfolders with images, or a video and accepts additional arguments to adjust the out directory and a frame gap. To view these arguments. 

```Shell
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
                  directoryis specified, 'results' directory will be created
                  at the same level as the input directory.
  -batch BATCH    Batch input. Path to a directory that contains sub-folders
                  containing images to test.
  -vid VID        Path to the video file to be converted. Include the video
                  itself. If inputing a video, the video willbe split into
                  subfolders to conserve memory when running uNLC.
  -fgap FGAP      Frame gap between images in a sequence. Default 0.
```

Note that there are three different methods for inputing data and two additional arguments which correspond to an output directory a frame gap. The frame gap is important when specifying the frames that uNLC is to compute across in a given image sequence. If a frame gap too small is chosen, memory errors may occur and the computation may take a long time. When using uNLC, it is important to be aware of the length of image sequences and the frame gap as to prevent memory issues. If a frame gap too large is chosen, the obtained results will be useless. Furthermore, uNLC works best on image sequences where there is significant motion by the foreground objects. If the objects are barely moving, or move relatively slow, it may be beneficial to increase the frame gap. The output directory is where the segmented images will be saved. If this is not specified, a results directory will be created at the same level of any given input. Two examples of use are shown below.

Example 1.) Segment a \textbf{short} video that is located at home/.../vidpath/video.avi. No output directory will be specified. A frame gap of 3 is desired. The commands and potential output should then be similar to the following.
```Shell
$ cd ISS_uNLC/
$ python iss_main.py -vid home/.../vidpath/video.avi -fgap 3

Video to Image Conversion: [ ##.# %]

Batch:  0
Input Directory:  /home/.../vidpath/src_images/00
Output Directory:  /home/.../vidpath/results
Loading images: [ ##.# %]
Total Sequence Shape:  (25, 512, 640, 3)
Memory Usage for Sequence: 24.58 MB.
24576000

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

Removing low energy blobs finished: 0.25 s

Batch:  1
Input Directory:  /home/.../vidpath/src_images/00
Output Directory:  /home/.../vidpath/results
Loading images: [ ##.# %]
Total Sequence Shape:  (25, 512, 640, 3)
Memory Usage for Sequence: 24.58 MB.
24576000
...
```

Once the process is complete (it may take several minutes depending on the length of the video), two new folders (\texttt{src\_images/} and \texttt{results/}) should have been created where the video is located. One contains the frames from the video (\texttt{src\_images/}), the other contains the segmentation results (\texttt{results/}). It is not recommend to use a video that is longer than 30 seconds.

Example 2.) Perform segmentation using the \texttt{-batch} input. In the context of \texttt{iss\_uNLC.py} script, this option is to be selected if there is a folder, containing several subfolders that contain images. This is illustrated below.
```Shell
- batch_folder/
-- image_directory_00/
--- img_00_00
--- img_00_01
--- ...
-- image_directory_01/
--- img_01_00
--- img_01_01
--- ...
-- ...
```

No output directory will be specified. A frame gap of 3 is desired. The commands and potential output should then be similar to the following.



\begin{lstlisting}
# Parameters for uNLC and pyflow.
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
\end{lstlisting}
