
import csv
import json
import matplotlib.pyplot as plt
import os
import cv2
from engine.BaseDataset import BaseDataset
import numpy as np

import logging
import pandas as pd
# # Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Drawer(BaseDataset):

    def __init__(self, args):
        super().__init__(args)


    def draw_AI_result_to_images(self):
        """
        Processes a CSV file to extract and visualize AI results by overlaying information onto images.

        Steps:
        1. Initialize empty lists for storing frame IDs and distances.
        2. Open and read the CSV file specified by `self.csv_file_path`.
        3. For each row in the CSV:
        - Convert the row to a string and locate the JSON data.
        - Parse the JSON data to extract frame information.
        - For each frame ID:
            a. Construct the image file path using the frame ID.
            b. Read the image using OpenCV.
            c. Overlay frame ID and other AI results onto the image:
                - Tail objects (`tailingObj`)
                - Detected objects (`detectObj`)
                - Vanishing line objects (`vanishLineY`)
                - ADAS objects (`ADAS`)
                - Lane information (`LaneInfo`)
            d. Optionally display the image with overlays and save the image to disk.
        - Handle exceptions related to JSON decoding and other unexpected errors.
        4. If enabled, plot distances to the camera over frame IDs using Matplotlib.

        Attributes:
        - `self.frame_ids`: List of frame IDs extracted from the CSV.
        - `self.distances`: List of distances corresponding to the tailing objects in each frame.
        - `self.csv_file_path`: Path to the CSV file containing AI results.
        - `self.image_basename`: Base name for image files.
        - `self.image_format`: Format of the image files.
        - `self.im_dir`: Directory where image files are located.
        - `self.show_tailingobjs`: Flag to indicate whether to draw tailing objects.
        - `self.show_detectobjs`: Flag to indicate whether to draw detected objects.
        - `self.show_vanishline`: Flag to indicate whether to draw vanish lines.
        - `self.show_adasobjs`: Flag to indicate whether to draw ADAS objects.
        - `self.show_laneline`: Flag to indicate whether to draw lane lines.
        - `self.show_airesultimage`: Flag to control whether to display the processed images.
        - `self.resize`: Flag to indicate whether to resize images before display.
        - `self.resize_w`, `self.resize_h`: Dimensions for resizing images.
        - `self.sleep`, `self.sleep_onadas`, `self.sleep_zeroonadas`: Time to wait before closing the image window.
        - `self.save_airesultimage`: Flag to control whether to save the processed images.
        - `self.save_imdir`: Directory to save the processed images.
        - `self.plot_label`: Label for the distance plot.
        - `self.show_distanceplot`: Flag to control whether to display the distance plot.

        Returns:
        - `frame_ids`: List of frame IDs processed.
        - `distances`: List of distances to the camera for the tailing objects.
        """
        frame_ids = []
        distances = []

        with open(self.csv_file_path, 'r') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                # Debug print for each row
                # print(f"Row: {row}")
                # Join the row into a single string
                row_str = ','.join(row)
                
                # Find the position of 'json:'
                json_start = row_str.find('json:')
                if json_start != -1:
                    json_data = row_str[json_start + 5:].strip()
                    if json_data.startswith('"') and json_data.endswith('"'):
                        json_data = json_data[1:-1]  # Remove enclosing double quotes
                    
                    # Replace any double quotes escaped with backslashes
                    json_data = json_data.replace('\\"', '"')
                    
                    try:
                        data = json.loads(json_data)
                        # print(f"Parsed JSON: {data}")

                        for frame_id, frame_data in data['frame_ID'].items():
                            
                            self.frame_ids.append(int(frame_id))
                            logging.info(f"frame_id:{frame_id}")
                            logging.info(f"csv_file_path:{self.csv_file_path}")

                            # Get image path
                            im_file = self.image_basename + frame_id + "." + self.image_format
                            im_path = os.path.join(self.im_dir,im_file)
                            logging.info(im_path)
                            im = cv2.imread(im_path)

                            cv2.putText(im, 'frame_ID:'+str(frame_id), (10,10), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 255, 255), 1, cv2.LINE_AA)
                            tailing_objs = frame_data.get('tailingObj', [])
                            vanish_objs = frame_data.get('vanishLine', [])
                            ADAS_objs = frame_data.get('ADAS', [])
                            detect_objs = frame_data.get('detectObj', {})
                            lane_info = frame_data.get("LaneInfo",[])

                            #---- Draw tailing obj----------
                            if tailing_objs and self.show_tailingobjs:
                                self.draw_tailing_obj(tailing_objs,im)
                            else:
                                self.distances.append(float('nan'))  # Handle missing values

                            # ------Draw detect objs---------
                            if detect_objs and self.show_detectobjs:
                                self.draw_detect_objs(detect_objs,im)                                                   

                            # -------Draw vanish line--------
                            if vanish_objs and self.show_vanishline:
                                self.draw_vanish_objs(vanish_objs,im)

                            # -------Draw ADAS objs-----------------
                            if ADAS_objs and self.show_adasobjs:
                                self.draw_ADAS_objs(ADAS_objs,im)

                            # Draw lane lines if LaneInfo is present
                            if lane_info and lane_info[0]["isDetectLine"] and self.show_laneline:
                                self.draw_laneline_objs(lane_info,im)

                            if self.show_airesultimage:
                                # 按下任意鍵則關閉所有視窗
                                if self.resize:
                                    im = cv2.resize(im, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
                                cv2.imshow("im",im)
                                if self.ADAS_FCW==True or self.ADAS_LDW==True:
                                    if self.sleep_zeroonadas:
                                        cv2.waitKey(0)
                                    else:
                                        cv2.waitKey(self.sleep_onadas)
                                else:
                                    cv2.waitKey(self.sleep)
                                # cv2.destroyAllWindows()
                            if self.save_airesultimage:
                                self.img_saver.save_image(im,frame_ID=frame_id)
                                # os.makedirs(self.save_imdir,exist_ok=True)
                                # im_file = self.image_basename + str(frame_id) + "." + self.image_format
                                # save_im_path = os.path.join(self.save_imdir,im_file)
                                # if not os.path.exists(save_im_path):
                                #     cv2.imwrite(save_im_path,im)
                                # else:
                                #     logging.info(f'image exists :{save_im_path}')
                                
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON: {e}")
                    except Exception as e:
                        logging.error(f"Unexpected error: {e}")
        
        if self.show_distanceplot:
            # Plotting the data
            plt.figure(figsize=(200, 100))
            plt.plot(self.frame_ids, self.distances, label=self.plot_label)

            plt.xlabel('FrameID')
            plt.ylabel('tailingObj.distanceToCamera')
            plt.title('Distance to Camera over Frames')
            plt.legend()
            plt.grid(True)

            plt.show()


        return frame_ids, distances
    


    def draw_tailing_obj(self,tailing_objs,im):
        """
        Draws bounding boxes and labels for tailing objects on an image.

        Parameters:
        - `tailing_objs`: A list containing dictionaries with details of tailing objects. Each dictionary includes:
            - 'tailingObj.distanceToCamera': The distance of the object from the camera.
            - 'tailingObj.id': The unique identifier of the tailing object.
            - 'tailingObj.x1': The x-coordinate of the top-left corner of the bounding box.
            - 'tailingObj.y1': The y-coordinate of the top-left corner of the bounding box.
            - 'tailingObj.x2': The x-coordinate of the bottom-right corner of the bounding box.
            - 'tailingObj.y2': The y-coordinate of the bottom-right corner of the bounding box.
            - 'tailingObj.label': The label of the tailing object (e.g., 'VEHICLE').

        - `im`: The image on which the bounding boxes and labels will be drawn.

        Steps:
        1. Extract details (distance to camera, ID, coordinates, and label) from the first tailing object in the list.
        2. Update instance variables `self.tailingObj_x1` and `self.tailingObj_y1` with the coordinates of the tailing object.
        3. Set parameters for drawing the bounding box:
            - `text_thickness`: Thickness of the text used for labels.
            - `color`: Color for the bounding box and text, which varies based on the distance to the camera.
            - `thickness`: Thickness of the bounding box lines.
        4. If `self.showtailobjBB_corner` is `True`, draw bounding box corners using lines instead of a rectangle:
            - Draw lines at each corner of the bounding box with varying thickness and color based on the distance to the camera.
        5. If `self.showtailobjBB_corner` is `False`, draw a standard rectangle bounding box around the object.
        6. Draw text labels above the bounding box:
            - Display the object label and ID.
            - Display the distance to the camera.
        7. Append the distance to `self.distances` list. If the distance is not available, append `float('nan')`.

        Attributes:
        - `self`: The instance of the class this method belongs to. Used to access instance variables like `self.showtailobjBB_corner` and `self.distances`.

        Returns:
        - None. The image `im` is modified in place with the bounding box and labels drawn on it.
        """
        distance_to_camera = tailing_objs[0].get('tailingObj.distanceToCamera', None)
        tailingObj_id = tailing_objs[0].get('tailingObj.id', None)
        tailingObj_x1 = tailing_objs[0].get('tailingObj.x1', None)
        tailingObj_y1 = tailing_objs[0].get('tailingObj.y1', None)
        tailingObj_x2 = tailing_objs[0].get('tailingObj.x2', None)
        tailingObj_y2 = tailing_objs[0].get('tailingObj.y2', None)
        
        tailingObj_label = tailing_objs[0].get('tailingObj.label', None)

        self.tailingObj_x1 = tailingObj_x1
        self.tailingObj_y1 = tailingObj_y1

        text_thickness = 0.45
        # Draw bounding box on the image
        if self.showtailobjBB_corner:
            top_left = (tailingObj_x1, tailingObj_y1)
            bottom_right = (tailingObj_x2, tailingObj_y2)
            top_right = (tailingObj_x2,tailingObj_y1)
            bottom_left = (tailingObj_x1,tailingObj_y2) 
            BB_width = abs(tailingObj_x2 - tailingObj_x1)
            BB_height = abs(tailingObj_y2 - tailingObj_y1)
            divide_length = 5
            thickness = 3
            color = (0,255,255)

            if distance_to_camera>=10:
                color = (0,255,255)
                thickness = 3
                text_thickness = 0.40
            elif distance_to_camera>=7 and distance_to_camera<10:
                color = (0,100,255)
                thickness = 5
                text_thickness = 0.46
            elif distance_to_camera<7:
                color = (0,25,255)
                thickness = 7
                text_thickness = 0.50

            # Draw each side of the rectangle
            cv2.line(im, top_left, (top_left[0]+int(BB_width/divide_length), top_left[1]), color, thickness)
            cv2.line(im, top_left, (top_left[0], top_left[1] + int(BB_height/divide_length)), color, thickness)

            cv2.line(im, bottom_right,(bottom_right[0] - int(BB_width/divide_length),bottom_right[1]), color, thickness)
            cv2.line(im, bottom_right,(bottom_right[0],bottom_right[1] - int(BB_height/divide_length) ), color, thickness)


            cv2.line(im, top_right, ((top_right[0]-int(BB_width/divide_length)), top_right[1]), color, thickness)
            cv2.line(im, top_right, (top_right[0], (top_right[1]+int(BB_height/divide_length))), color, thickness)

            cv2.line(im, bottom_left, ((bottom_left[0]+int(BB_width/divide_length)), bottom_left[1]), color, thickness)
            cv2.line(im, bottom_left, (bottom_left[0], (bottom_left[1]-int(BB_height/divide_length))), color, thickness)
        else:
            cv2.rectangle(im, (tailingObj_x1, tailingObj_y1), (tailingObj_x2, tailingObj_y2), color=(0,255,255), thickness=2)


        # if tailingObj_label=='VEHICLE':
            # Put text on the image
        # if not self.show_detectobjs:
        cv2.rectangle(im,(tailingObj_x1, tailingObj_y1-10),(tailingObj_x2 , tailingObj_y1-10),(50,50,50), -1)
        cv2.putText(im, f'{tailingObj_label} ID:{tailingObj_id}', (tailingObj_x1, tailingObj_y1-10), cv2.FONT_HERSHEY_SIMPLEX, text_thickness, color, 1, cv2.LINE_AA)
    
        cv2.putText(im, 'Distance:' + str(round(distance_to_camera,3)) + 'm', (tailingObj_x1, tailingObj_y1-25), cv2.FONT_HERSHEY_SIMPLEX,text_thickness+0.1, color, 1, cv2.LINE_AA)

        if distance_to_camera is not None:
            self.distances.append(distance_to_camera)
        else:
            self.distances.append(float('nan'))  # Handle missing values

    
    def draw_detect_objs(self,detect_objs,im):
        """
        Draws detected objects with bounding boxes and labels on an image.

        Parameters:
        - `detect_objs`: A dictionary where each key represents an object type and the corresponding value is a 
        list of objects detected. Each object is a dictionary containing keys:
            - 'detectObj.label': The label of the detected object (e.g., 'VEHICLE', 'HUMAN').
            - 'detectObj.x1': The x-coordinate of the top-left corner of the bounding box.
            - 'detectObj.y1': The y-coordinate of the top-left corner of the bounding box.
            - 'detectObj.x2': The x-coordinate of the bottom-right corner of the bounding box.
            - 'detectObj.y2': The y-coordinate of the bottom-right corner of the bounding box.
            - 'detectObj.confidence': The confidence score of the detection.
        - `im`: The image on which the detected objects will be drawn.

        Steps:
        1. Iterate over each object type and its list of detected objects in `detect_objs`.
        2. For each detected object:
        - Extract the label, bounding box coordinates (x1, y1, x2, y2), and confidence score.
        - Skip drawing the bounding box if it matches the specified tailing object coordinates (`self.tailingObj_x1` and `self.tailingObj_y1`).
        - Otherwise, draw the bounding box around the detected object using a specific color based on the label:
            - `VEHICLE`: Orange color.
            - `HUMAN`: Light orange color.
        - Add the label and confidence score text above the bounding box.

        Attributes:
        - `self`: The instance of the class this method belongs to. Used to access the instance variables `self.show_tailingobjs`, `self.tailingObj_x1`, and `self.tailingObj_y1`.

        Returns:
        - None. The image `im` is modified in place with the bounding boxes and labels drawn on it.
        """
        # Draw detectObj bounding boxes
        for obj_type, obj_list in detect_objs.items():
            for obj in obj_list:
                label = obj.get(f'detectObj.label', '')
                x1 = obj.get(f'detectObj.x1', 0)
                y1 = obj.get(f'detectObj.y1', 0)
                x2 = obj.get(f'detectObj.x2', 0)
                y2 = obj.get(f'detectObj.y2', 0)
                confidence = obj.get(f'detectObj.confidence', 0.0)
                
                if self.show_tailingobjs and self.tailingObj_x1==x1 and self.tailingObj_y1==y1:
                    # Draw bounding box
                    continue
                else:
                    # Draw bounding box
                    if label == "VEHICLE":
                        color=(255,150,0)
                    elif label=="HUMAN":
                        color=(0,128,255)
                    cv2.rectangle(im, (x1, y1), (x2, y2), color=color, thickness=1)
                    cv2.putText(im, f'{label} {confidence:.2f}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    def draw_vanish_objs(self,vanish_objs,im):
        vanishlineY = vanish_objs[0].get('vanishlineY', None)
        logging.info(f'vanishlineY:{vanishlineY}')
        x2 = im.shape[1]
        cv2.line(im, (0, vanishlineY), (x2, vanishlineY), (0, 255, 255), thickness=1)
        cv2.putText(im, 'VanishLineY:' + str(round(vanishlineY,3)), (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.45, (0, 255, 255), 1, cv2.LINE_AA)


    def draw_ADAS_objs(self,ADAS_objs,im):
        self.ADAS_FCW = ADAS_objs[0].get('FCW',None)
        self.ADAS_LDW = ADAS_objs[0].get('LDW',None)
        logging.info(f'ADAS_FCW:{self.ADAS_FCW}')
        logging.info(f'ADAS_LDW:{self.ADAS_LDW}')
        if self.ADAS_FCW==True:
            cv2.putText(im, 'Collision Warning', (150,50), cv2.FONT_HERSHEY_SIMPLEX,1.3, (0, 128, 255), 2, cv2.LINE_AA)
        if self.ADAS_LDW==True:
            cv2.putText(im, 'Departure Warning', (150,80), cv2.FONT_HERSHEY_SIMPLEX,1.3, (128, 0, 255), 2, cv2.LINE_AA)

    def draw_laneline_objs(self,lane_info,im):
        """
        Draws lane lines on an image based on lane information.

        Parameters:
        - `lane_info`: A list of dictionaries containing lane line information. The dictionary should have keys 
        "pLeftCarhood.x", "pLeftCarhood.y", "pLeftFar.x", "pLeftFar.y", "pRightCarhood.x", "pRightCarhood.y", 
        "pRightFar.x", and "pRightFar.y", representing points on the left and right sides of the lane.
        - `im`: The image on which the lane lines will be drawn.

        Steps:
        1. Extract points from the `lane_info` for left and right carhood and far points.
        2. Compute the width of the lane at the carhood and far points.
        3. Calculate the main lane points by shifting the left and right carhood and far points towards the center.
        4. Create an array of points defining the lane polygon and another for the main lane polygon.
        5. Create an overlay image and fill the main lane polygon with a green color.
        6. Blend the overlay with the original image using a transparency factor.
        7. Optionally draw the polygon border and direction line (commented out in the code).
        8. Draw the left and right lane lines on the image:
        - Left lane line: Drawn in blue.
        - Right lane line: Drawn in red.

        Attributes:
        - `self`: The instance of the class this method belongs to. Not used directly in this method but included for context.

        Returns:
        - None. The image `im` is modified in place with the lane lines drawn on it.
        """
        pLeftCarhood = (lane_info[0]["pLeftCarhood.x"], lane_info[0]["pLeftCarhood.y"])
        pLeftFar = (lane_info[0]["pLeftFar.x"], lane_info[0]["pLeftFar.y"])
        pRightCarhood = (lane_info[0]["pRightCarhood.x"], lane_info[0]["pRightCarhood.y"])
        pRightFar = (lane_info[0]["pRightFar.x"], lane_info[0]["pRightFar.y"])

        width_Cardhood = abs(pRightCarhood[0] - pLeftCarhood[0])
        width_Far = abs(pRightFar[0] - pLeftFar[0])

        pLeftCarhood_mainlane = (pLeftCarhood[0]+int(width_Cardhood/4.0),pLeftCarhood[1])
        pLeftFar_mainlane = (pLeftFar[0]+int(width_Far/4.0),pLeftFar[1])
        pRightCarhood_mainlane = (pRightCarhood[0]-int(width_Cardhood/4.0),pRightCarhood[1])
        pRightFar_mainlane = (pRightFar[0]-int(width_Far/4.0),pRightFar[1])               
        # Create an array of points to define the polygon
        points = np.array([pLeftCarhood, pLeftFar, pRightFar, pRightCarhood], dtype=np.int32)
        points_mainlane = np.array([pLeftCarhood_mainlane,
                                    pLeftFar_mainlane,
                                    pRightFar_mainlane,
                                    pRightCarhood_mainlane], dtype=np.int32)
        # Reshape points array for polylines function
        points = points.reshape((-1, 1, 2))

        # Create an overlay for the filled polygon
        overlay = im.copy()
        cv2.fillPoly(overlay, [points_mainlane], color=(0, 255, 0))  # Green filled polygon

        # Blend the overlay with the original image
        alpha = self.alpha  # Transparency factor
        cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0, im)

        # Optionally, draw the polygon border
        # cv2.polylines(image, [points], isClosed=True, color=(0, 0, 0), thickness=2)  # Black border

        # Draw for direction
        # pmiddleFar_mainlane = (int((pLeftFar[0]+pRightFar[0])/2.0),int((pLeftFar[1]+pRightFar[1])/2.0))
        # pmiddleCarhood_mainlane = (int((pLeftCarhood[0]+pRightCarhood[0])/2.0),int((pLeftCarhood[1]+pRightCarhood[1])/2.0))
        # cv2.line(image, pmiddleFar_mainlane, pmiddleCarhood_mainlane, (0, 255, 255), 1)  # Blue line

        # Draw left lane line
        cv2.line(im, pLeftCarhood, pLeftFar, (255, 0, 0), self.laneline_thickness)  # Blue line
        # Draw right lane line
        cv2.line(im, pRightCarhood, pRightFar, (0, 0, 255), self.laneline_thickness)  # Red line