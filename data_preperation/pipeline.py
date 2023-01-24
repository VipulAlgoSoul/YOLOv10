from __future__ import print_function, division
import os
import ast

import cv2
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

from .create_directory import CreateDirectory
from .pipeline_help import get_data_to_pipe
from configs.getconfig import GetConfig
from .visualize import draw_from_imtxt, get_cords_from_yolo, display_image


class Pipeline(Dataset):

    '''This class contains methods which finally gives out dataloader.
    It also stores samples of train data in new dir crated'''

    def __init__(self, data_cnf_path):
        '''Initializes the pipeline class, create exp path dictionary and data path dict'''
        init_confg = GetConfig(data_cnf_path)
        init_config = init_confg()
        self.exp_path = init_config['PATH']['exp_path']
        self.data_path = init_config['PATH']['data_folder']
        self.image_shape = ast.literal_eval(init_config['DATA']['image_shape'])
        self.grids = int(init_config['DATA']['grids'])
        self.per_grid = int(init_config['DATA']['per_grids'])
        self.grid_jump = int(init_config['DATA']['grid_jump'])
        self.slider_aspect = bool(int(init_config['DATA']['slider_aspect']))
        self.recursive_grid_flag = bool(int(init_config['DATA']['use_recursive_grids']))
        self.recursive_pergrid_flag = bool(int(init_config['DATA']['use_recursive_pergrids']))

        self.debug_pipeline = bool(int(init_config['DEBUG']['pipeline']))
        self.debug_pipeline_help = bool(int(init_config['DEBUG']['pipeline_help']))
        self.visualize_inp = bool(int(init_config['VISUALIZE']['input_visualize']))

        self.exp_path_dict = CreateDirectory(self.exp_path)
        self.data_path_dict = self.data_path_seperate(self.data_path)


    def __call__(self):

        self.gip = get_data_to_pipe(grids=self.grids, per_grids=self.per_grid, imshape=self.image_shape,
                                    train_path=self.data_path_dict["train_path"], grid_jump = self.grid_jump,
                                    slider_aspect= self.slider_aspect, recursive_grids=self.recursive_grid_flag,
                                    recursive_pergrids= self.recursive_pergrid_flag,
                                    save_path=self.exp_path_dict["results"], debug=self.debug_pipeline_help)

        self.dict_cord_units, self.config_save_dict, self.data_save_dict  = self.gip.do()

        if self.visualize_inp:
            self.input_visualize()

        for imp, im_an in zip(self.data_save_dict["Image Path"], self.data_save_dict["Annotation pipeline"]):
            im_ann = ast.literal_eval(im_an)
            img = cv2.imread(imp).copy()
            img = cv2.resize(img, self.image_shape)
            print(im_ann)

            single_cord_dict =self.collect_single(im_ann)
            print(single_cord_dict)

            if self.visualize_inp:
                self.display_singledict(single_cord_dict, img, "Compare orig and IOU")


    def display_singledict(self,single_cord_dict, img, title ="window"):

        '''

        This class method displays Oringinal bounding box and corresponding IOU anchor
        box in an image.

        :param single_cord_dict: IOU anchor box collected from collect_single class method.
        :param img: corresponding input image.
        :param title: name of the image window.
        :return: None

        '''

        image = img.copy()
        for key, val in single_cord_dict.items():
            key_cnv = ast.literal_eval(key)
            val_crd = val[-1]
            val_cnf = val[0]
            key_crd = get_cords_from_yolo(1,1,key_cnv)
            val_crd = get_cords_from_yolo(1,1,val_crd)

            cv2.rectangle(image, key_crd[1], key_crd[2], (0,255,0), 2)
            cv2.rectangle(image, val_crd[1], val_crd[2], (0,0,255), 2)

            display_image(image, title=title)


    def input_visualize(self):

        '''

        This class method displays images and thier class bounding boxes.

        :return: None
        This class method is just to visualize the input and the class bounding boxes.

        '''

        vis_cnt = 3 # this count is th enumber of image displayed
        image_files = self.data_save_dict["Image Path"]
        text_files = self.data_save_dict["Text Path"]
        anot_fl = self.data_save_dict["Annotation pipeline"]
        class_maps = self.config_save_dict['class map']

        for img, txt in zip(image_files[0:vis_cnt], text_files[0:vis_cnt]):
            draw_from_imtxt(img, txt, title="Train Samples")

        for img, ann in zip(image_files[0:vis_cnt], anot_fl[0:vis_cnt]):
            im = cv2.imread(img).copy()
            im = cv2.resize(im, self.image_shape)
            for an in ast.literal_eval(ann):
                crds = get_cords_from_yolo(1,1,an)
                class_name = class_maps[crds[0]]
                cv2.rectangle(im, crds[1], crds[2], (0, 0, 0) , 3)
                cv2.putText(im, class_name, crds[1],cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255, 0, 0), thickness=2)
            display_image(im,title="Annotation for train" )



    def data_path_seperate(self, data_path):

        '''

        This class method gives out path dictionary.

        :param data_path: input parent folder path where the train, test and validate data is to be looked for.
        :return: path dictionary with train_path, test_path and validate_path.

        '''

        if not os.path.exists(data_path):
            raise Exception("The path to data doesnot exist")
        data_path_dict ={}
        data_path_dict["train_path"] = os.path.join(data_path, "train")
        data_path_dict["test_path"] = os.path.join(data_path,"test")
        data_path_dict["validate_path"] = os.path.join(data_path,"validate")

        return data_path_dict

    def collect_single(self, box_crds):

        '''

        This class method returns a dictionary of annotated bounding box as key and its
        corresponding highest IOU anchor box as value.

        :param box_crds: The coordinates of class bounding boxes, output from pipeline_help.
        :return: IOU based anchor coordinates.

        '''

        crd_dict ={}
        for crds in box_crds:
            ann_cord = get_cords_from_yolo(1,1, crds)
            crd_temp_dict ={}
            for key, val in self.dict_cord_units.items():
                key_chann_dict ={}
                for grid_crdes in val:
                    grid_crds =[None]
                    grid_crds.extend(grid_crdes)
                    anchr_crds = get_cords_from_yolo(1,1,grid_crds)
                    iou_val = self.IOU_cords(ann_cord, anchr_crds)
                    key_chann_dict[iou_val] = grid_crds
                if self.debug_pipeline:
                    print("Cord Annotate, Grid channel \n",key_chann_dict)
                max_cnf = max(list(key_chann_dict.keys()))
                max_crd = key_chann_dict[max_cnf]
                crd_temp_dict[max_cnf] = (key, max_crd)
            if self.debug_pipeline:
                print("The final channel and corddict \n", crd_temp_dict)
            maxx_select = max(list(crd_temp_dict.keys()))
            crd_dict[str(crds)] = [maxx_select, crd_temp_dict[maxx_select][0], crd_temp_dict[maxx_select][1]]
        if self.debug_pipeline:
            print("Final cord text to channel out \n", crd_dict)

        return crd_dict




    def IOU_cords(self, annot_crd, anchor_crd):

        '''

        This class methods calculates the IOU between two coordinates.

        :param annot_crd: Cordinate of single class bounding box.
        :param anchor_crd: Archor cordinate currently looked into by algorithm.
        :return: IOU score.

        '''

        annot_x_set = set(np.arange(annot_crd[1][0], annot_crd[2][0]))
        annot_y_set = set(np.arange(annot_crd[1][1], annot_crd[2][1]))

        anchor_x_set = set(np.arange(anchor_crd[1][0], anchor_crd[2][0]))
        anchor_y_set = set(np.arange(anchor_crd[1][1], anchor_crd[2][1]))

        x_inter = annot_x_set.intersection(anchor_x_set)
        y_inter = annot_y_set.intersection(anchor_y_set)

        area_inter = len(x_inter)*len(y_inter)

        x_union = annot_x_set.union(anchor_x_set)
        y_union = annot_y_set.union(anchor_y_set)

        area_union = len(x_union)*len(y_union)

        return area_inter/area_union

