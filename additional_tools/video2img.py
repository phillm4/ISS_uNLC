"""
Author:     Mitchell Phillips
File:       video2img.py
Date:       December 2017
Purpose:    Convert frames of video file to a directory of images. 
"""

import cv2
import numpy as np

import argparse
import errno
import os
import sys


def vid_valid(vid):
    """
    Validate the input video file.
    """

    if (not os.path.isfile(vid) or 
        not vid.lower().endswith(('.mp4','.avi'))):
        raise argparse.ArgumentTypeError(
            'Cannot read video file. If the video type cannot be read, ' \
            'check this source code if extension is listed as valid type.')
    
    return(vid)


def command_line_parse():
    """
    Parse and check the command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('vid', type = vid_valid,
    help = 'Path to the video file to be converted. Include the video itself.')
    args = parser.parse_args()

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


def convert_video(vid_path,save_frames,images_per_batch):
    """
    Convert a video file to a directory of images. Can specify the 
    number of frames between each recorded image. 
    INPUT:  vid_path - Path to video file including the video file 
            itself.
            save_frames - Specifies the frames to save. Is like a 
            frame gap. However a value of 1 will save every frame, 2
            will save every other, 3 saves every third frame, etc. 
            images_per_batch - Number of images to be saved in each 
            batch.  
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
        print('Error loading video. Check inputs or ffMPEG install.')

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


def main():
    """
    Convert frames of video file to a directory of images.
    INPUT: vid_path - Path to video file.
    OUTPUT: [] - System out. Directory of images taken from video. 
    """

    args = command_line_parse()
    vid_path = args.vid
    convert_video(vid_path,1,100)


if __name__ == "__main__":
    main()