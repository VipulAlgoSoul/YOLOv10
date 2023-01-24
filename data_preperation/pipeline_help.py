import os
import json
import math

import numpy as np
import cv2
import pandas as pd

from .utils import *

class get_data_to_pipe:
    """get data from  folder, create anchor boxes, analyse data"""

    def __init__(self, grids, per_grids, imshape , train_path , grid_jump ,
                 slider_aspect , recursive_grids, recursive_pergrids, save_path,debug=True):
        self.grids = grids
        self.per_grids = per_grids
        self.imshape = imshape
        self.debug = debug
        self.train_path = train_path
        self.grid_jump = grid_jump
        self.slider_aspect = slider_aspect
        self.recursive_grids = recursive_grids
        self.recursive_pergrids = recursive_pergrids
        self.save_path = save_path
        self.df_pipehelp = {"Image Path": [], "Text Path":[], "Class Count Dict": [],
                            "Annotation Original" :[], "Annotation pipeline": []}

    def do(self):
        self.collect_image_gridcrds()
        self.get_output_arrangements()
        self.pipeline_txt()

        return self.dict_cord_units, self.save_lines, self.df_pipehelp

    def collect_image_gridcrds(self):
        '''

        Split the image inot grids, split grids into level regions.

        :return: dictionary of regions
        '''

        if len(self.imshape)>2:
            self.row, self.col, self.chl = self.imshape
        elif len(self.imshape)==2:
            self.row, self.col = self.imshape
        else:
            raise Exception("The imshape should contain atleast two dimensions")

        # The following True if condition creates a list of creatable grids
        if self.recursive_grids:
            grid_list = [1]
            grid_list.extend(list(np.arange(2,self.grids+1, self.grid_jump))) #recursively make grids from 1 to self.grids
        else:
            grid_list = [self.grids]

        grid_dictionary = {}
        for grd in grid_list:
            grd_cord, channel_out = self.grid_iterate(grd) # creates grd_cord and channel out
            grid_dictionary[grd] = grd_cord

        self.grid_dict = grid_dictionary
        self.channel_dict = channel_out

        # print("\n")
        # if self.debug:
        #     print(channel_out, "The channel out" ,"\n")
        #     print("the Grid Dictionary ",grid_dictionary)


    def grid_iterate(self, grd):

        rw_wid = self.row/grd
        cl_wid = self.col/grd

        aspect_ratios, channel_dict = self.get_bbox_aspects()
        #

        centre_crds = self.collect_all_grid_centres(rw_wid=rw_wid, cl_wid=cl_wid)

        rw_mul = rw_wid/self.per_grids
        cl_mul = cl_wid/self.per_grids

        grid_dict = {}
        for aspect in aspect_ratios:
            grid_dict[aspect] = [[i[0], i[1], rw_mul*aspect[0], cl_mul*aspect[1]] for i in centre_crds.copy()]

        if self.slider_aspect:
            grid_dict = self.extend_aspect(grid_dict)

        if self.debug:
            for key , val_list in grid_dict.items():
                dr_im = np.zeros(self.imshape, dtype=np.uint8)
                for val in val_list:
                    dr_im = cv2.circle(dr_im, (int(val[0]), int(val[1])), 2, (255,255,255), -1)
                    dr_im = cv2.rectangle(dr_im, (int(val[0]-val[2]/2), int(val[1]-val[3]/2)), (int(val[0]+val[2]/2), int(val[1]+val[3]/2)),(255,255,255),2)
                cv2.imshow(str(key), dr_im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return grid_dict, channel_dict

    def extend_aspect(self, grid_dict):

        main_perkey = (self.per_grids, self.per_grids)
        grid_val = grid_dict[main_perkey][0]

        xhalf_wid = grid_val[2] / 2
        yhalf_wid = grid_val[3] / 2

        upd_dict ={}
        for aspect_key, asp_list in grid_dict.items():
            if aspect_key != main_perkey:
                if aspect_key[0] < aspect_key[1]:
                    for sin_crd in asp_list:
                        x1 = sin_crd[0] - xhalf_wid
                        y1 = sin_crd[1] - yhalf_wid

                        num_slides_x = math.ceil(sin_crd[3] / sin_crd[2])
                        slide_width_x = sin_crd[3] / num_slides_x
                        slidex_half = slide_width_x // 2

                        for i in range(1, num_slides_x+1):
                            new_key = (10*aspect_key[0]+i, aspect_key[1])
                            if new_key not in upd_dict.keys():
                                upd_dict[new_key]=[]
                            upd_dict[new_key].append([int(x1+ slide_width_x*i - slidex_half), int(sin_crd[1]), int(slide_width_x), int(sin_crd[3])])

                else:
                    for sin_crd in asp_list:
                        x1 = sin_crd[0] - xhalf_wid
                        y1 = sin_crd[1] - yhalf_wid

                        num_slides_y = math.ceil(sin_crd[2] / sin_crd[3])
                        slide_width_y = sin_crd[2] / num_slides_y
                        slidey_half = slide_width_y // 2

                        for i in range(1, num_slides_y + 1):
                            new_key = (aspect_key[0], 10 * aspect_key[1] + i)
                            if new_key not in upd_dict.keys():
                                upd_dict[new_key] = []
                            upd_dict[new_key].append([int(sin_crd[0]), int(y1 + i * slide_width_y - slidey_half), int(sin_crd[2]),int(slide_width_y)])

        grid_dict.update(upd_dict)

        return grid_dict


    def collect_all_grid_centres(self, rw_wid, cl_wid):
        centre_crd =[]
        for row in np.arange(0,self.imshape[0], rw_wid):
            rc = row+rw_wid/2
            for col in np.arange(0, self.imshape[1], cl_wid):
                clc = col+cl_wid/2
                centre_crd.append((rc,clc))
        return centre_crd

    def get_bbox_aspects(self):
        '''collects order of arrangements'''
        out_channel ={}
        aspect_ratios = []
        cnt= 0
        if self.recursive_pergrids:
            asp_seq = np.arange(2, self.per_grids+1)
            for per_grid_seq in asp_seq:
                # cnt = 0
                for i in np.arange(1, per_grid_seq + 1):
                    aspect_ratios.append((i, per_grid_seq))
                    out_channel[cnt] = (i, per_grid_seq)
                    cnt+=1
                for i in np.arange(1, per_grid_seq):
                    aspect_ratios.append((per_grid_seq, i))
                    out_channel[cnt] = (per_grid_seq,i)
                    cnt+=1

        else:
            # cnt = 0
            for i in np.arange(1, self.per_grids + 1):
                aspect_ratios.append((i, self.per_grids))
                out_channel[cnt] = (i, self.per_grids)
                cnt+=1
            for i in np.arange(1, self.per_grids):
                aspect_ratios.append((self.per_grids, i))
                out_channel[cnt] = (self.per_grids,i)
                cnt+=1

        return aspect_ratios, out_channel

    def get_output_arrangements(self):
        '''This function creates output formats
        4 for cordinates xc, yc, w, h
        n for classes +1 background class
        data_path can be temp datapath'''



        #collect class text : find number of classes
        if "classes.txt" not in os.listdir(self.train_path):
            raise Exception("classes.txt file not exist")
        class_path = os.path.join(self.train_path, "classes.txt")
        class_lines = read_txt(class_path)
        n_class = len(class_lines)
        self.class_count_dict = {i:0 for i in class_lines}
        self.class_count_dict["bg"]=0
        self.class_map_dict = {i: cls for i, cls in enumerate(self.class_count_dict.keys())}
        total_op_length = n_class+1+4

        #find intersection and IOU
        self.collect_data_frame()
        # corddict arrange and visualize



        if self.debug:
            print(self.class_map_dict, "class lines")
            print(self.class_count_dict, "class counts")
            # print(self.df_pipehelp)

        #do augmentation
        #create dataframe


    def collect_data_frame(self):
        '''GEts IOU and collect the regions'''
        dict_len ={}
        dict_cord_units = {}
        for layers in self.grid_dict.values():
            # through grid dict items
            for key, val in layers.items():
                if key not in dict_len.keys():
                    dict_len[key] = 0
                    dict_cord_units[key] = []
                dict_len[key] = dict_len[key]+len(list(val))
                dict_cord_units[key].extend(val)

        self.num_channels = len(dict_len)
        #check all values are same
        val_set = set(dict_len.values())
        if len(val_set)!=1:
            raise Exception("Not same out channel dimensions")

        self.pseudo_channel_len = val_set.pop()

        # print(num_channels, pseudo_channel_len, len(dict_cord_units))
        # print(dict_cord_units)
        self.dict_cord_units = dict_cord_units

        im_list = [i for i in os.listdir(self.train_path) if os.path.splitext(i)[-1] != ".txt"]
        imtxt_dict = {os.path.join(self.train_path,i):os.path.join(self.train_path,os.path.splitext(i)[0]+".txt") for i in im_list}

        for im_p, txt_p in imtxt_dict.items():
            cordlist, dict_cnt, ann_orig =self.get_single(text_path=txt_p)
            #add to class count dict
            # self.df_pipehelp = {"Image Path": [], "Text Path": [], "Class Count Dict": [],
            #                     "Annotation Original ": [], "Annotation pipeline": [], "Grid IOU Target": []}

            self.df_pipehelp["Image Path"].append(im_p)
            self.df_pipehelp["Text Path"].append(txt_p)
            self.df_pipehelp["Class Count Dict"].append(str(dict_cnt))

            for clss, cnt in dict_cnt.items():
                cls = self.class_map_dict[int(clss)]
                self.class_count_dict[cls] = self.class_count_dict[cls]+cnt

            self.df_pipehelp["Annotation Original"].append(str(ann_orig))
            self.df_pipehelp["Annotation pipeline"].append(str(cordlist))
            # self.df_pipehelp[""].append([])
            # self.df_pipehelp[""].append()
        df = pd.DataFrame(self.df_pipehelp)
        df.to_excel(os.path.join(self.save_path,"data.xlsx"))
        # print(self.df_pipehelp)

    def get_single(self, text_path):
        text_lines, class_dict , ann_orig= collect_cords_txt(txt_path=text_path,imshape=self.imshape)
        # print("\n",text_lines)
        # print(class_dict)
        return text_lines, class_dict, ann_orig

    def pipeline_txt(self):
        self.save_lines = {"train directory": self.train_path,
                      "image shape": self.imshape,
                      "class map": self.class_map_dict,
                      "class counts": self.class_count_dict,
                      "grids": self.grids,
                      "per grid": self.per_grids,
                      "number output channels":self.num_channels,
                      "per out channel length without class multiplication": self.pseudo_channel_len,
                      }

        #save as json file
        with open(os.path.join(self.save_path,"data_analyse.txt"), 'w') as f:
            for key, value in self.save_lines.items():
                f.write('%s:%s\n' % (key, value))





# gip = get_data_to_pipe(grids=8,per_grids=3,imshape=(500,500),train_path="D:\Detection_Classification\data//train")
# gip.do()

