import cv2
import matplotlib.pyplot as plt
import csv
import json
import os
from PIL import Image, ImageDraw
import logging
import numpy as np
from utils.saver import ImageSaver
import glob
import yaml
from config.args import Args
import colorlog
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseDataset:
    def __init__(self,args):

        # self.logging = logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # Input settings
        self.im_dir = args.im_dir
        self.im_folder = None
        self.image_basename = args.image_basename
        self.csv_file_path = args.csv_file
        self.csv_file = os.path.basename(self.csv_file_path)
        self.image_format = args.image_format


        # Enable / Disable save AI result images
        self.save_airesultimage = args.save_airesultimage
        self.save_jsonlog = args.save_jsonlog
        self.save_rawimages = args.save_rawimages

        # How fast of show the images
        self.sleep = args.sleep
        self.sleep_zeroonadas = args.sleep_zeroonadas
        self.sleep_onadas = args.sleep_onadas

        # Enable / disable plot frame-distance
        self.distances = []
        self.frame_ids = []

        # Enable / disable show objs on images
        self.show_airesultimage = args.show_airesultimage
        self.show_detectobjs = args.show_detectobjs
        self.show_tailingobjs = args.show_tailingobjs
        self.show_vanishline = args.show_vanishline
        self.show_adasobjs = args.show_adasobjs
        self.showtailobjBB_corner = args.showtailobjBB_corner
        self.show_laneline = args.show_laneline
        self.show_distancetitle = args.show_distancetitle
        self.show_detectobjinfo = args.show_detectobjinfo
        self.show_devicemode = args.show_devicemode

        # Lane line
        self.alpha = args.alpha
        self.laneline_thickness = args.laneline_thickness

        self.tailingobjs_BB_thickness = args.tailingobjs_BB_thickness
        self.tailingobjs_BB_colorB = args.tailingobjs_BB_colorB
        self.tailingobjs_BB_colorG = args.tailingobjs_BB_colorG
        self.tailingobjs_BB_colorR = args.tailingobjs_BB_colorR
        self.tailingobjs_text_size = args.tailingobjs_text_size
        self.tailingobjs_distance_decimal_length = args.tailingobjs_distance_decimal_length

        self.tailingObj_x1 = None
        self.tailingObj_y1 = None

        self.ADAS_FCW = False
        self.ADAS_LDW = False

        # Enable/Disable show customer resized images
        self.resize = args.resize
        self.resize_w = args.resize_w
        self.resize_h = args.resize_h

        #csv file list
        self.csv_file_list = ['assets/csv_file/golden_date_ImageMode_10m.csv',
                                'assets/csv_file/golden_date_ImageMode_20m.csv',
                                'assets/csv_file/golden_date_ImageMode_30m.csv',
                                'assets/csv_file/golden_date_ImageMode_40m.csv',
                                'assets/csv_file/golden_date_ImageMode_50m.csv',]
        self.list_label = ['GT_10m',
                           'GT_20m',
                           'GT_30m',
                           'GT_40m',
                           'GT_50m']
        
        #plot label
        self.plot_label = args.plot_label

        self.mode = args.mode

        # if not self.mode=='eval' and not self.mode=='evaluation' and self.save_rawimages:
        self.img_saver = ImageSaver(args)
        # else:
        #     self.img_saver = None


        # Video extract frames parameters
        self.skip_frame = 10
        self.crop = True
        self.crop_top = 0.3
        self.crop_left = 0.1
        self.crop_right = 0.9

        self.model_w = args.model_w
        self.model_h = args.model_h


    def display_parameters(self):
        logging.info("--------------- 📊 BaseDataset Settings 📊 ---------------")
        
        logging.info(f"🗂️  IMAGE DIRECTORY          : {self.im_dir}")
        logging.info(f"🖼️  IMAGE BASE NAME          : {self.image_basename}")
        logging.info(f"📂 CSV FILE PATH            : {self.csv_file_path}")
        logging.info(f"📄 CSV FILE                 : {self.csv_file}")
        logging.info(f"🖼️  IMAGE FORMAT             : {self.image_format}")

        
        logging.info("------------- 💾 SAVE SETTINGS ---------------------")
        logging.info(f"💾 SAVE AI RESULT IMAGE     : {self.save_airesultimage}")
        logging.info(f"💾 SAVE JSON LOG            : {self.save_jsonlog}")
        logging.info(f"💾 SAVE RAW IMAGES          : {self.save_rawimages}")

        logging.info("------------- 💤 SLEEP SETTINGS ---------------------")
        logging.info(f"💤 SLEEP                   : {self.sleep}")
        logging.info(f"💤 SLEEP ZERO ON ADAS      : {self.sleep_zeroonadas}")
        logging.info(f"💤 SLEEP ON ADAS           : {self.sleep_onadas}")

        logging.info("------------- 📺 DISPLAY SETTINGS ---------------------")
        logging.info(f"🖼️  SHOW AI RESULT IMAGE    : {self.show_airesultimage}")
        logging.info(f"🔍 SHOW DETECT OBJS         : {self.show_detectobjs}")
        logging.info(f"🚗 SHOW TAILING OBJS        : {self.show_tailingobjs}")
        logging.info(f"🧩 SHOW VANISH LINE         : {self.show_vanishline}")
        logging.info(f"🚘 SHOW ADAS OBJS           : {self.show_adasobjs}")
        logging.info(f"📏 SHOW TAIL OBJ BB CORNER  : {self.showtailobjBB_corner}")
        logging.info(f"🛣️  SHOW LANE LINE          : {self.show_laneline}")
        logging.info(f"📍 SHOW DISTANCE TITLE      : {self.show_distancetitle}")

        logging.info("------------- 🚗 LANE LINE SETTINGS ---------------------")
        logging.info(f"🚗 LANE LINE ALPHA          : {self.alpha}")
        
        logging.info("------------- 📏 TAILING OBJS SETTINGS ---------------------")
        logging.info(f"📏 TAILING OBJS BB THICKNESS : {self.tailingobjs_BB_thickness}")
        logging.info(f"🎨 TAILING OBJS BB COLOR (B, G, R): ({self.tailingobjs_BB_colorB}, {self.tailingobjs_BB_colorG}, {self.tailingobjs_BB_colorR})")
        logging.info(f"🔠 TAILING OBJS TEXT SIZE   : {self.tailingobjs_text_size}")
        logging.info(f"🔍 TAILING OBJ X1           : {self.tailingObj_x1}")
        logging.info(f"🔍 TAILING OBJ Y1           : {self.tailingObj_y1}")

        logging.info("------------- 🚗 ADAS SETTINGS ---------------------")
        logging.info(f"🚗 ADAS FCW                 : {self.ADAS_FCW}")
        logging.info(f"🚗 ADAS LDW                 : {self.ADAS_LDW}")
        
        logging.info("------------- 📐 RESOLUTION SETTINGS ---------------------")
        logging.info(f"📐 RESIZE                  : {self.resize}")
        logging.info(f"📏 RESIZE WIDTH            : {self.resize_w}")
        logging.info(f"📏 RESIZE HEIGHT           : {self.resize_h}")

        logging.info("------------- 📄 CSV FILE LIST SETTINGS ---------------------")
        logging.info(f"📄 CSV FILE LIST           : {self.csv_file_list}")
        logging.info(f"🏷️  LIST LABELS             : {self.list_label}")
        
        logging.info("------------- 📊 PLOT LABEL SETTINGS ---------------------")
        logging.info(f"📊 PLOT LABEL              : {self.plot_label}")

        logging.info("------------- 🎞️  VIDEO EXTRACT FRAMES SETTINGS ---------------------")
        logging.info(f"⏯️  VIDEO SKIP FRAME        : {self.skip_frame}")
        
        logging.info("------------- ✂️  CROP SETTINGS ---------------------")
        logging.info(f"✂️  CROP                   : {self.crop}")
        logging.info(f"⬆️  CROP TOP               : {self.crop_top}")
        logging.info(f"⬅️  CROP LEFT              : {self.crop_left}")
        logging.info(f"➡️  CROP RIGHT             : {self.crop_right}")

        logging.info("------------- 🛠️  MODEL SETTINGS ---------------------")
        logging.info(f"🛠️  MODEL WIDTH            : {self.model_w}")
        logging.info(f"🛠️  MODEL HEIGHT           : {self.model_h}")
        
        logging.info("------------------------------------------------------------")




    def visualize(self, mode=None, 
                    jsonlog_from=None, 
                    plot_distance=False, 
                    gen_raw_video=False,
                    save_raw_image_dir=None, 
                    extract_video_to_frames=None, 
                    crop=False, 
                    raw_images_dir=None):
            '''
            Main function for visualizing AI results. Handles different operational modes and utility tasks.

            Args:
                mode (str, optional): Operational mode. Options include "online", "semi-online", "offline".
                jsonlog_from (str, optional): Source of JSON logs. Options include "camera" and "online".
                plot_distance (bool, optional): If True, plots distance data.
                gen_raw_video (bool, optional): If True, generates a video from raw images.
                save_raw_image_dir (str, optional): Directory to save raw images.
                extract_video_to_frames (str, optional): Path to video for frame extraction.
                crop (bool, optional): If True, applies cropping to extracted frames.
                raw_images_dir (str, optional): Directory containing raw images for video generation.
            '''
            return NotImplemented
    

    def visualize_online(self):
        '''
        Visualizes AI results in real-time by transferring JSON logs and raw images from the camera to the local computer for display.

        Details:
            - Transfers JSON logs and JPEG images from the camera.
            - Visualizes the data on the local computer.
        '''
        return NotImplemented
    

    def visualize_semi_online(self):
        '''
        Visualizes AI results in a semi-online mode by first transferring raw images and then processing the JSON logs.

        Details:
            - Transfers raw images from the camera to the local computer.
            - Runs online mode to transfer and visualize JSON logs on saved images.
        '''
        return NotImplemented
    

    def visualize_offline_jsonlog_from_camera(self):
        '''
        Visualizes AI results offline using JSON logs saved from the camera.

        Details:
            - Processes JSON logs saved in a CSV file format.
        '''
        return NotImplemented
    
    def visualize_offline_jsonlog_from_online(self):
        '''
        Visualizes AI results offline using JSON logs saved during online mode.

        Details:
            - Processes JSON logs obtained from previous online visualizations.
        '''
        return NotImplemented
    

    def start_server(self):
        '''
        Starts a server to receive and process JSON logs from the camera.

        Details:
            - Listens for incoming JSON logs from the camera.
            - Parses the logs to draw AI results on raw images.
        '''
        return NotImplemented
    

    def start_server_ver2(self):
        '''
        Starts a server to receive frame index, image data, and JSON logs from the camera.

        Details:
            - Listens for frame index, image size, and image data.
            - Saves the received image and processes the JSON log to draw AI results.
        '''
        return NotImplemented
    
    def execute_remote_command_with_progress(self, command):
        '''
        Executes a command on the LI80 camera via SSH and reports progress.

        Args:
            command (str): Command to be executed on the remote camera.
        '''
        return NotImplemented
    

    def execute_local_command(self, command):
        '''
        Executes a command on the local computer.

        Args:
            command (str): Command to be executed locally.
        '''
        return NotImplemented
    

    def receive_image_and_log(self, client_socket):
        '''
        Receives image data and JSON logs from a client socket.

        Details:
            - Receives frame index, image size, image data, and JSON log.
            - Saves the image and processes the JSON log.
        
        Args:
            client_socket (socket.socket): Socket connection to the client.
        '''
        return NotImplemented
    
    def process_json_log(self, json_log):
        '''
        Processes a single frame's JSON log to visualize AI results.

        Args:
            json_log (dict): JSON log data for one frame.
        '''
        return NotImplemented
    
    def draw_AI_result_to_images(self):
        '''
        Processes a CSV file containing JSON logs and draws AI results on the corresponding raw images.
        '''
        return NotImplemented
    

    def draw_bounding_boxes(self, frame_ID, tailing_objs, detect_objs, vanish_objs, ADAS_objs, lane_info):
        '''
        Draws bounding boxes and other annotations on the image based on the provided JSON data.

        Args:
            frame_ID (int): Identifier for the frame.
            tailing_objs (list): List of tailing objects to annotate.
            detect_objs (dict): Dictionary of detected objects to annotate.
            vanish_objs (list): List of vanish line objects to annotate.
            ADAS_objs (list): List of ADAS objects to annotate.
            lane_info (dict): Information about lane markings and related data.
        '''
        return NotImplemented
    



    def extract_distance_data(self,csv_file):
        return NotImplemented
    
    def plot_distance_value_on_each_frame_ID(self):
        return NotImplemented

    def compare_distance_in_two_csv_file(self):
        return NotImplemented
    
    def compare_distance_in_multiple_csv_file(self):
        return NotImplemented

    def draw_AI_result_to_images(self):
        return NotImplemented
    
  

    

    '''
    ------------------------------------------
    FUNC: process_json_log
        input : json_log only one frame
        Purpose : Process just one frame JSON log
    ------------------------------------------
    '''
    def process_json_log(self,json_log):
        return NotImplemented
    


    
    
            
    