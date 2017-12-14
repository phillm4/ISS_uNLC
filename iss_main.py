"""
Author:     Mitchell Phillips
File:       iss_main.py
Date:       December 2017
Purpose:    Main program to run the uNLC algorithm. The program is 
essentially a wrapper around Pathak's nlc algorithm and is heavily 
inspired by the NLC library run_full.py and nlc.py scripts with 
several differences, both stylistically and functionally. This script
is intended to be used to perform instrument shaft segmentation on 
robotic surgery images, however it may be used on other types of 
images if desired. 

Outline:    
        - Import additional libraries.
        - Argparser functions.
        - Set up folders / initialize directories.
        - uNLC functions.
        - Main execution.
"""

import cv2
import numpy as np

import argparse
import errno
import os
import sys
import time

import _init_nlc_path
import videoseg.src.nlc as uNLC
import videoseg.src.utils as utils


def directory_valid(directory):
    """
    Validate that the directory is a legitimate location.
    """
    if not directory or not os.path.exists(directory):
        raise argparse.ArgumentTypeError('Please enter a valid directory.')

    return(directory)


def vid_valid(vid_path):
    """
    Validate the input video file.
    """

    if (not os.path.isfile(vid_path) or 
        not vid_path.lower().endswith(('.mp4','.avi'))):
        raise argparse.ArgumentTypeError(
            'Cannot read video file. If the video type cannot be read, ' \
            'check this source code if extension is listed as valid type.')
    
    return(vid_path)



def command_line_parse():
    """
    Parse and check the command line arguments. Arguments similar to 
    those required for Pathak's nlc.py. Slight modifications and 
    additions to provide readability and usability for endoscopic 
    images and videos. 
    """

    parser = argparse.ArgumentParser(
        description='Wrapper for Pathak\'s Non-Local Consensus Voting '\
            'algorithm for Instrument Shaft Segmentation.')
    
    parser.add_argument('-indir',
        help = 'Path to the input directory that contains the images to test.',
        default = False,
        type = directory_valid)

    parser.add_argument('-outdir',
        help = 'Path to the output directory to save results. If no directory' \
            'is specified, \'results\' directory will be created at the same ' \
            'level as the input directory.',
        default = False,
        type = directory_valid)

    parser.add_argument('-batch',
        help = 'Batch input. Path to a directory that contains sub-folders ' \
            'containing images to test.',
        default = False,
        type = directory_valid)

    parser.add_argument('-vid',
        help = 'Path to the video file to be converted. ' \
            'Include the video itself. If inputing a video, the video will' \
            'be split into subfolders to conserve memory when running uNLC.',
        default = False,
        type = vid_valid)
    
    parser.add_argument('-fgap',
        help = 'Frame gap between images in a sequence. Default 0.',
        default = 0, 
        type = int)
      
    args = parser.parse_args()

    if not args.indir and not args.vid and not args.batch:
        print('Input images or video required. Exit program. See \'[-h].')
        sys.exit(0)

    return args


def make_directory(new_directory):
    """
    Make a new folder in the current working directory. Check if the 
    folder to be created already exists.
    INPUT:  new_directory - Folder to be created. 
    OUTPUT: [] - System out. New directory.
    """
    try:
        os.makedirs(new_directory)
    except OSError as err:
        if err.errno == errno.EEXIST:
            pass
        else:
            raise 


def convert_video(vid_path,save_frames):
    """
    Convert a video file to a directory of images. Can specify the 
    number of frames between each recorded image. 
    INPUT:  vid_path - Path to video file including the video file 
            itself.
            save_frames - Specifies the frames to save. Is like a 
            frame gap. However a value of 1 will save every frame, 2
            will save every other, 3 saves every third frame, etc.  
    OUTPUT: vid_dir - Path to where the video is stored. This is not 
            the path to the video file itself. It is the folder the 
            video is located in.
            save_dir - Path to where the converted images are stored.
            [] - System out. Directory of images taken from video. 
    """
    
    # Specify the recorded frame parameters.
    start_frame = 0
    i = 0

    # Create new folder to store images. New folder will be located 
    # where the video is stored.
    vid_dir = os.path.dirname(vid_path)
    save_dir = os.path.join(vid_dir, 'src_images')
    make_directory(save_dir)      

    vidcap = cv2.VideoCapture(vid_path)
    if not vidcap.isOpened():
        print('Error loading video. Check inputs.')

    # Estimated total number of frames in video. 
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))    
    
    # Save specified frames for the entire duration of the video.
    notdone = True
    while notdone:
        notdone, frame = vidcap.read()
        
        if ((notdone and (i-start_frame)%(save_frames) == 0) or 
            (notdone and i == start_frame)):
            sub_save_dir = '/' + str(i//100).zfill(2)

            make_directory(save_dir+sub_save_dir)
    
            save_name = (save_dir  + sub_save_dir + '/frame_' 
                + str(i).zfill(5) + '.jpg')
            cv2.imwrite(save_name,frame)
            
            # Print status.
            sys.stdout.write('Video to Image Conversion: [% .1f%%]\r' %(
                (i/video_length)*100))
            sys.stdout.flush()
        
        i+=1 

    return vid_dir, save_dir


def initialize_batch(img_directory,batch_directory,vid_path):
    """
    Set up a list containing the paths to each of the sub-directories 
    containing the image sequences.
    INPUT:  img_directory - Path to directory containing an image 
            sequence.
            batch_directory - Path to a directory containing 
            sub-directories of images. 
            vid_path - Path to video file which will be converted to 
            bathes of image sequences. 
    OUTPUT: batch_directory - Path to a directory containing 
            sub-directories of images. 
            batch - List containing the paths to each of the 
            sub-directories in the batch_directory. 
    """

    if img_directory:
        batch = [os.path.abspath(img_directory)]
        batch_directory = os.path.abspath(img_directory)

    # Convert video to images if video path was specified. 
    else:
        
        if vid_path:
            vid_path = os.path.abspath(vid_path)
            vid_directory, batch_directory = convert_video(vid_path,1)
        
        batch_directory = os.path.abspath(batch_directory)
        batch = os.listdir(batch_directory)
        batch = [batch_directory +'/'+ name for name in batch]
        batch = [name for name in batch if os.path.isdir(name)]
        batch.sort()


        if batch == []:
            print('Could not form batches. Check if input is an individual ' 
                'image directory and use \'-indir\' instead of \'-batch\'.') 

    return batch_directory, batch


def initialize_output_directory(batch_directory, out_directory):
    """
    Create output directory if none was declared.
    INPUT:  batch_directory - Path to directory containing 
            sub-directories containing images.
            out_directory - Output directory where results are to be 
            saved to.  
    OUTPUT: out_directory - Path to output directory where results 
            will be saved. 
    """

    if out_directory == False:
        out_directory = os.path.join(
            os.path.dirname(batch_directory), 'results')
        make_directory(out_directory)

    out_directory = os.path.abspath(out_directory)

    return out_directory


def images_from_directory(directory):
    """
    Find all the image files with '.jpg' or '.JPEG' in the directory.
    INPUT:  directory - Directory containing the images of interest. 
    OUTPUT: img_list - List containing the file names of all the 
            images with specified extensions in the directory.
    """

    os.chdir(directory)
    img_list = os.listdir('./')
    img_list = [name for name in img_list if name.lower().endswith(
        ('.jpg','.jpeg','.png'))]
    img_list.sort()

    if len(img_list) < 2:
        print('Not enough images found. Need more than one. End program.')
        sys.exit(0)
    
    return img_list


def load_images(img_list, img_resize):
    """
    Load images from a given list. Note, if using pytorch or another 
    deeplearning framework, may need to check the axis order. For 
    example, in pytorch the axis are to be swapped from numpy (HxWxC) 
    to torch (CxHxW).
    INPUT:  img_list - List containing file path to images.
            Resize - Dimensions that images are to be resized. 
            Tuple as (height, width).
    OUTPUT: data - n x H x W x C numpy array of image data. 
    """
    
    data = np.zeros(
        ((len(img_list),img_resize[1],img_resize[0],3)),dtype=np.uint8)
    
    for j, img in enumerate(img_list):
        
        # Pathak uses PIL and skimage functions which use RGB color
        # space rather than OpenCV's BRG.
        img = cv2.imread(img)[:,:,::-1]
        data[j,:,:,:] = cv2.resize(img, img_resize)

        # Print status.
        sys.stdout.write('Loading images: [% .1f%%]\r' %(
            ((j+1)/len(img_list))*100))
        sys.stdout.flush()

    return data         


def iss_uNLC(img_directory,out_directory,frame_gap,batch_number):
    """
    Perform uNLC segmentation on a given image sequence. After 
    initializing parameters, an image sequence is constructed from 
    the images contained in a given directory and a specified 
    frame gap value. This sequence is stored as a numpy array and 
    as a result, memory issues may occur if too long of a sequence is
    attempted to be segmented. A check is in place to prevent this
    sequences exceeding a certain memory threshold are skipped.
    
    INPUT:  img_directory - Path to folder contain image sequence.
            out_directory - Path to folder where segmented images 
            will be saved.
            frame_gap - The number of images to be skipped between 
            images in a sequence. This parameter helps free up memory 
            if the sequence is very long. In addition, if there is not
            much motion, it may be beneficial to increase this.
            batch_number - Indicator for the current batch of images
            undergoing uNLC segmentation.
    
    OUTPUT: [] - System out. uNLC statistics printed to the terminal 
            and segmented images are saved to the out_directory.  
    """

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

    # Display image sequence details.
    print('\nBatch: ', batch_number)
    print('Input Directory: ', img_directory)
    print('Output Directory: ', out_directory)

    # Construct image sequence. 
    img_list = images_from_directory(img_directory)
    img_list = img_list[0:len(img_list):frame_gap + 1]
    height, width, color_depth = (cv2.imread(img_list[0])).shape
    height, width = [int(height*resize_fraction), int(width*resize_fraction)]
    image_sequence = load_images(img_list, (width, height))
    sequence_length = image_sequence.shape[0]
    sequence_memory = float(image_sequence.nbytes) / 1e+6

    # Image sequence statistics.
    print('Total Sequence Shape: ', image_sequence.shape)
    print('Memory Usage for Sequence: %.2f MB.' %sequence_memory)

    print(image_sequence.nbytes)

    if sequence_memory > memory_limit:
        print('*****Warning: Image sequence may be too large!*****\n'
            'Consider changing the sequence parameters.')

    
    print('\n*****Performing uNLC*****\n')
    uNLC_sequence = uNLC.nlc(image_sequence, maxsp=max_superpixels, 
        iters=vote_iterations, pyflow_parameters=pyflow_parameters,
        outdir=out_directory, clearBlobs=clearVoteBlobs,
        binTh=segmentation_energy_threshold, relEnergy=False,
        dosave=False)
    
    # Post processing.
    print('\n*****uNLC complete. Save results.*****\n')
    if clearFinalBlobs:
        uNLC_sequence = uNLC.remove_low_energy_blobs(
            uNLC_sequence, segmentation_energy_threshold)

    # Place a colored mask over-top a grayscale image and save this 
    # as the segmentation results. May need to change this up when 
    # it is desired to use the images to train a CNN / FCN. 
    for i in range(sequence_length):
        mask = (uNLC_sequence[i]>segmentation_energy_threshold).astype(np.uint8)
        img_gray = cv2.cvtColor(image_sequence[i],cv2.COLOR_RGB2GRAY)
        img_masked = np.zeros(image_sequence[i].shape, dtype=np.uint8)
        
        for c in range(3):
            img_masked[:, :, c] = img_gray / 2 + 127
        
        img_masked[mask.astype(np.bool), 1:] = 0
        img_name = out_directory+'/seg_'+img_list[i]
        cv2.imwrite(img_name, img_masked[:,:,::-1])


def main():
    """
    INPUT:  sys.argv[1] - Requires path to either an image directory,
            directory containing subfolders with images, or a video.
            sys.argv[2] - Optional argument for the directory where
            segmented images will be saved to.
            sys.argv[3] - Optional argument controlling the frame gap
            between images in a given image sequence.
    OUTPUT: [] - System out. Statistics to printed to the terminal.
            Segmented images saved to either given or created results
            directory.
    """
    
    # System inputs.
    cwd = os.getcwd()
    args = command_line_parse() 
    img_directory = args.indir
    out_directory = args.outdir
    batch_directory = args.batch
    vid_path = args.vid
    frame_gap = args.fgap

    # Initialize the image paths and setup output directory.
    batch_directory, batch = initialize_batch(
        img_directory,batch_directory,vid_path)
    out_directory = initialize_output_directory(batch_directory, out_directory) 

    # Iterate through the given image subdirectories and perform the 
    # uNLC segmentation. Save results to the output directory. 
    # for i, batch_subset in enumerate(batch):
    #     iss_uNLC(batch_subset,out_directory,frame_gap,i)


if __name__ == "__main__":
    main()
