import csv
import json
import matplotlib.pyplot as plt
import os
import cv2
from engine.BaseDataset import BaseDataset
import numpy as np
# from config.config import get_connection_args
import logging
import pandas as pd

from utils.drawer import Drawer
# # Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Visualize(BaseDataset):
    def __init__(self, args):
        """
        Initializes the Historical dataset object.

        Args:
            args (argparse.Namespace): Arguments containing configuration parameters such as directories and connection settings.
        """
        super().__init__(args)
        self.camera_rawimages_dir = args.camera_rawimages_dir
        self.camera_csvfile_dir = args.camera_csvfile_dir
        self.current_dir = os.getcwd()  # Current working directory (e.g., .../Historical)
        self.im_folder = os.path.basename(self.im_dir)

        # self.display_parameters()
        self.Drawer = Drawer(args)
       
        super().display_parameters()

    def visualize(self, mode=None, jsonlog_from=None):
        """
        Main function for visualizing AI results based on the specified mode and options.

        This function controls the overall flow of visualization depending on the mode selected.
        It can operate in offline mode only (For Jeremy)

        Args:
            mode (str, optional): 
                The operational mode to run the visualization in. Options include:
                - "offline": Process pre-recorded data without real-time inputs.
                           
            jsonlog_from (str, optional): 
                Specifies the source of the JSON logs when running in offline mode. Options include:
                - "camera": Use JSON logs generated from camera data.

        Function Flow:
            - If `mode` is "offline", the function processes JSON logs from "camera" sources.
           
        """
        if mode == "offline":
            if jsonlog_from == "camera":
                logging.info("🎥 Running offline mode with JSON log from camera...")
                self.visualize_offline_jsonlog_from_camera()
          

    def visualize_offline_jsonlog_from_camera(self):
        """
        📷 Visualizes AI results offline using JSON logs saved from the camera.
        Downloads raw images and CSV file if not present locally.
        """
        logging.info("🖼️ Drawing AI results on images...")
        self.Drawer.draw_AI_result_to_images()
        logging.info("🎉 Visualization complete!")



    
   
            

    


    