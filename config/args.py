import yaml
import os

# Argument parsing for class initialization
class Args:
    def __init__(self, config):

        # Model configuation
        self.model_w = config['MODEL']['INPUT_WIDTH']
        self.model_h = config['MODEL']['INPUT_HEIGHT']

        # Mode
        self.mode = config['MODE']['VISUALIZE_MODE']


        # JSON log
        self.jsonlog_from = config['JSONLOG']['FROM']


        # Local configuration
        self.im_dir = config['LOCAL']['RAW_IMG_DIR']
        self.csv_file = config['LOCAL']['CSV_FILE']


        # Image configuration
        self.image_basename = config['IMAGE']['BASE_NAME']
        self.image_format = config['IMAGE']['FORMAT']

        # Display settings
        self.show_distanceplot = config['DISPLAY']['SHOW_DISTANCE_PLOT']
        self.show_airesultimage = config['DISPLAY']['SHOW_AI_RESULT_IMAGE']
        self.show_detectobjs = config['DISPLAY']['SHOW_DETECT_OBJS']
        self.show_tailingobjs = config['DISPLAY']['SHOW_TAILING_OBJS']
        self.show_vanishline = config['DISPLAY']['SHOW_VANISH_LINE']
        self.show_adasobjs = config['DISPLAY']['SHOW_ADAS_RESULT']
        self.showtailobjBB_corner = config['DISPLAY']['SHOW_TAILING_OBJS_BB_CORNER']
        self.show_laneline = config['DISPLAY']['SHOW_LANE_INFO']
        self.show_distancetitle = config['DISPLAY']['SHOW_DISTANCE_TITLE']
        self.show_detectobjinfo = config['DISPLAY']['SHOW_DETECT_OBJS_INFO']
        self.show_devicemode = config['DISPLAY']['SHOW_DEVICE_MODE']


        # Lane line
        self.alpha = config['LANE_LINE']['ALPHA']
        self.laneline_thickness = config['LANE_LINE']['THICKNESS']

        # Tailing obj bounding box
        self.tailingobjs_BB_thickness = config['TAILING_OBJ']['BOUDINGBOX_THINKNESS']
        self.tailingobjs_BB_colorB = config['TAILING_OBJ']['BOUDINGBOX_COLOR_B']
        self.tailingobjs_BB_colorG = config['TAILING_OBJ']['BOUDINGBOX_COLOR_G']
        self.tailingobjs_BB_colorR = config['TAILING_OBJ']['BOUDINGBOX_COLOR_R']
        self.tailingobjs_text_size = config['TAILING_OBJ']['TEXT_SIZE']
        self.tailingobjs_distance_decimal_length = config['TAILING_OBJ']['DISTANCE_DECIMAL_LENGTH']

        # Resize settings
        self.resize = config['RESIZE']['ENABLED']
        self.resize_w = config['RESIZE']['WIDTH']
        self.resize_h = config['RESIZE']['HEIGHT']

        # Save settings
        self.save_rawimages = config['SAVE']['RAW_IMAGES']
        self.save_airesultimage = config['SAVE']['AI_RESULT_IMAGE']
        self.save_jsonlog = config['SAVE']['JSON_LOG']


        # Wait settings
        self.sleep = config['WAIT']['VALUE']
        self.sleep_zeroonadas = config['WAIT']['ZERO_ON_ADAS_EVENT']
        self.sleep_onadas = config['WAIT']['ON_ADAS_EVENT']

    
