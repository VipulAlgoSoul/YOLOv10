import os
import shutil

def CreateDirectory(parent_path):
    '''This function creates directory tree'''

    curr_path_dict ={}
    if not os.path.exists(parent_path):
        raise Exception("Parent Path Not Found")
    exp_in_dir = len(os.listdir(parent_path))

    fl = str(exp_in_dir+1)
    #create current exp folder
    os.mkdir(os.path.join(parent_path,fl))
    curr_exp_path = os.path.join(parent_path,fl)
    curr_path_dict["parent"] = curr_exp_path
    if not os.path.exists(curr_exp_path):
        raise Exception("Path Creation Failed")
    folders = ["models", "checkpoints", "results", "configs", "samples"]

    for i in folders:
        folder_path = os.path.join(curr_exp_path, i)
        os.mkdir(folder_path)
        curr_path_dict[i] = folder_path

    return curr_path_dict