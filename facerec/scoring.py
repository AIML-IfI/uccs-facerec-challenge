# this file contains functionality to compute similarity scores between enrolled templates and probe features, and to handle score files
import numpy as np
import os
from torch import load

def cosine(x, y):
    """
    It calculates the pairwise cosine similarity scores
    """
    # Normalize x and y along axis 1
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    # Compute the cosine similarity using NumPy einsum
    similarity = np.einsum("nc,ck->nk", x, y.T)

    return similarity

def create_score_file(enrollment,probe_path,result_file):
    """
    It writes the scores between enrollment and probe to the result file
    """
    subject_ids,gallery_enroll = enrollment

    with open(result_file, "w") as f:

        f.write("FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT," + ",".join(subject_ids)+"\n")

        for img in os.listdir(probe_path):

            #get all information including detection scores,bboxes,embeddings in that image
            probe_infos = load(os.path.join(probe_path,img))
            probe_detection_scores = probe_infos["detection_scores"]
            probe_bboxes = probe_infos["bboxes"]
            probe_embeddings = probe_infos["embeddings"]

            #cosine similarity scores
            cos_sim = np.round(cosine(probe_embeddings,gallery_enroll),4)

            for ind in range(len(cos_sim)):
                
                # write them to file
                img_name = img[:-4]
                dt_sc = probe_detection_scores[ind][0]
                x1,y1,w,h = probe_bboxes[ind]
            
                id_scores = [s for s in cos_sim[ind]]
                line = [img_name, str(dt_sc), str(x1), str(y1), str(w), str(h)]

                line.extend(map(str, id_scores))

                f.write(",".join(line) + "\n")
