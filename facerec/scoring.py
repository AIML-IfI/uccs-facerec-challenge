# this file contains functionality to compute similarity scores between enrolled templates and probe features, and to handle score files
import numpy as np
import os
import torch
from torch import load

def cosine(x,y, gpu_index=None):
    """
    It calculates the pairwise cosine similarity scores using PyTorch.
    Supports GPU usage.

    args:
        x: the matrix of probes (N,512)
        y: the matrix of all gallery templates (M,512)
        gpu_index: indice of the gpu to be used, for cpu it should be given 'None'
    """
    # Determine the device (CPU or GPU)
    device = torch.device("cuda:{}".format(gpu_index) if gpu_index is not None and torch.cuda.is_available() else "cpu")

    # Normalize gallery templates once and store it in a closure
    def normalize_y_once(y):
        y = torch.tensor(y, dtype=torch.float32, device=device)
        return torch.nn.functional.normalize(y, p=2, dim=1)

    if not hasattr(cosine, 'y_norm'):
        cosine.y_norm = normalize_y_once(y)

    # Convert input array x to PyTorch tensor and move it to the specified device
    x = torch.tensor(x, dtype=torch.float32, device=device)

    # Normalize x along axis 1
    x = torch.nn.functional.normalize(x, p=2, dim=1)

    # Compute the cosine similarity using PyTorch matmul
    similarity = torch.matmul(x, cosine.y_norm.t())

    return similarity.cpu().numpy() if gpu_index is not None else similarity.numpy()

def create_score_file(enrollment,probe_path,result_file,gpu_index = None):
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
            cos_sim = cosine(probe_embeddings,gallery_enroll,gpu_index)

            for ind in range(len(cos_sim)):
                
                # write them to file
                img_name = img[:-4]
                dt_sc = probe_detection_scores[ind][0]
                x1,y1,w,h = probe_bboxes[ind]
            
                id_scores = [s for s in cos_sim[ind]]
                line = [img_name, str(dt_sc), str(x1), str(y1), str(w), str(h)]

                line.extend(map(str, id_scores))

                f.write(",".join(line) + "\n")
