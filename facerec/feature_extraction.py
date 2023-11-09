# This file contains source code related to face normalization and feature extraction
import numpy as np
import torch.utils.data as data
import torch
import os

class ImgData(data.Dataset):
    """
    It takes data containing the paths of images, bounding boxes/landmarks to 
    """
    def __init__(self, data,which_set=None,image_size=112, align=True, transform=None):
        
        self.img_paths,bboxes,landmarks = data
        self.points = landmarks if align else bboxes

        self.which_set = which_set

        if align:
            assert self.points != None
            self.align = align
            self.mode = "arcface"
            self.arcface_src = np.array(
                                [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                [41.5493, 92.3655], [70.7299, 92.2041]],
                                dtype=np.float32)
            self.arcface_src = np.expand_dims(self.arcface_src, axis=0)

                                    
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

def save_features_information(data,img_names,which_set,features,save_directory,batch_size_perImg):
    """
    It saves all information including detection_score,bboxes,landmarks and embedding as .pth file
    """

    if which_set == "gallery":
        # save the all information based on the each identity
        for idx in range(0,batch_size_perImg,10):

            identity = img_names[idx].split("_")[0]
            landmarks = np.stack([data[name][1][0] for name in img_names[idx:idx+10]])

            image_values = {
                "landmarks": landmarks,
                "embeddings": features[idx:idx + 10, :]
            }
            
            torch.save(image_values,os.path.join(save_directory,f"{identity}.pth"))

    else:
        # otherwise, save the all information of the each image
        image_values = {
            "detection_scores": data[img_names[0]][0],
            "bboxes": data[img_names[0]][1],
            "landmarks": data[img_names[0]][2],
            "embeddings": features
        }

        torch.save(image_values,os.path.join(save_directory,f"{img_names[0][:-4]}.pth"))

