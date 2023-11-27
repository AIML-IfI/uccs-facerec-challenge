# This file contains the source code required for running the enrollment from the extracted features
import os
import numpy as np
from torch import load

def average(gallery_path):

    subject_ids = []
    gallery_embeds = []

    for idx,id in enumerate(sorted(os.listdir(gallery_path))):
        
        #get the the path of the id
        id_embedding_pth = os.path.join(gallery_path,id)
        
        #read the embeddings
        id_info = load(id_embedding_pth)
        id_embeddings = id_info["embeddings"]

        #take the avarage of all embeddings belonging to that id
        id_embendidng = np.mean(id_embeddings,axis=0)
            
        #otherwise, get all embeddings for that identity
        s_id = f"{idx+1:04d}"
        subject_ids.append(s_id)
        gallery_embeds.append(id_embendidng)

    return subject_ids,np.stack(gallery_embeds)