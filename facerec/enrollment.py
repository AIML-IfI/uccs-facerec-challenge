# This file contains the source code required for running the enrollment from the extracted features
import os
import numpy as np
from torch import load
from .dataset import read_ground_truth
import warnings

def average(data_dir,embeddings_dir,set="gallery"):

    data = read_ground_truth(data_dir,set)
    subject_ids = sorted(np.unique([ f'{v[1][0]:04d}' for v in data.values()]))

    gallery_embeds = []

    for id in subject_ids:
        
        #get the the path of the id
        id_embedding_pth = os.path.join(embeddings_dir,f"{id}.pth")

        #check the galllery embeddings if exists
        if not os.path.exists(id_embedding_pth):
            warnings.warn(f"The path {id_embedding_pth} does not exist.", category=Warning)
            continue

        #read the embeddings
        id_info = load(id_embedding_pth)
        id_embeddings = id_info["embeddings"]

        #take the avarage of all embeddings belonging to that id
        id_embedding = np.mean(id_embeddings,axis=0)
            
        #otherwise, get all embeddings for that identity
        gallery_embeds.append(id_embedding)

    return subject_ids,np.stack(gallery_embeds)