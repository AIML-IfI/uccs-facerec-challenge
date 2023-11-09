# This file contains functionality to read the protocol files
import pandas as pd
import os
import csv

def read_ground_truth(data_dir,which_set):
    """
    It reads the ground_truth .csv file that contains bboxes,subject_ids and/or landmarks in the images
    For the test set, it only reads the image names as keys of the dictionary; the values are ``None``.

    It returns a dictionary based on image keys and their values that contain subject id and bboxes/landmarks
    """
    csv_path = os.path.join(data_dir,which_set + ".csv")
    if not os.path.exists(csv_path):
        raise ValueError("The ground truth file '%s' does not exist" % which_set)
    
    csv_file = pd.read_csv(csv_path)
    img_files = csv_file["FILE"].unique()

    data = {}
    
    for img in img_files:

        ground_truth = csv_file[csv_file["FILE"] == img]
        #each image can have subject_id, bboxes and/or landmarks
        subject_id = ground_truth["SUBJECT_ID"].astype(int).tolist()

        if which_set == "gallery":
            #for gallery; ground truth is facial landmarks
            ground_tr = ground_truth[["REYE_X", "REYE_Y", "LEYE_X", "LEYE_Y", "NOSE_X", "NOSE_Y",
                            "RMOUTH_X", "RMOUTH_Y", "LMOUTH_X", "LMOUTH_Y"]].values.reshape(len(subject_id),5,2)
            face_id = None
        else:
            #for validation; ground truth is bounding boxes
            ground_tr = ground_truth[["FACE_X","FACE_Y","FACE_WIDTH","FACE_HEIGHT"]].values
            face_id = ground_truth["FACE_ID"].astype(int).tolist()
 
        data[img] = (face_id,subject_id,ground_tr)

    return data

def read_detections(csv_path):
    """
    It reads the .csv file that contains bboxes and/or landmarks of the detections in the images

    It returns a dictionary based on image keys and their values that contain detection score, bounding boxes and landmarks if they exist
    """
    if not os.path.exists(csv_path):
        raise ValueError("The score file '%s' does not exist" % csv_path)

    score_file = pd.read_csv(csv_path)

    data = {}

    for img,group in score_file.groupby(by="FILE"):

        #each image can have detection scores,bboxes and landmarks
        detection_scores = group[["DETECTION_SCORE"]].values
        bboxes = group[["BB_X","BB_Y","BB_WIDTH","BB_HEIGHT"]].values

        try:
            landmarks = group[["REYE_X", "REYE_Y", "LEYE_X", "LEYE_Y", "NOSE_X", "NOSE_Y",
                            "RMOUTH_X", "RMOUTH_Y", "LMOUTH_X", "LMOUTH_Y"]].values.reshape(-1,5,2)
        except:
            landmarks = None
        
        data[img] = (detection_scores,bboxes,landmarks)

    return data

def read_recognitions(score_file):
    """
    It reads the score file
    """
    data = {}
    with open(score_file) as data_file:
        # read data line by line
        reader = csv.reader(data_file)
        # Skip the header line
        next(reader)

        for row in reader:

            if not len(row):
            # skip empty lines
                continue
            image,detection_score,bbox, cosine_scores = _get_values(row)
            
            if image not in data:
                data[image] = ([], [], [])  # Initialize with empty lists for detection scores, boxes, and similarity scores

            data[image][0].append(detection_score)
            data[image][1].append(bbox)
            data[image][2].append(cosine_scores)

    return data

def _get_values(splits):
  
  # each line should has 1006 values according to the score file structure
  if len(splits) != 1006 :
    raise ValueError("Cannot interpret score file line '%s'" % ",".join(splits))

  image = splits[0] + ".jpg"
  detection_score = float(splits[1])
  bbx = [float(v) for v in splits[2:6]] # x, y, width, height
  scores = [float(v) for v in splits[6:]] # scores in order id1_score, id2_score, ...

  return image, detection_score, bbx,scores