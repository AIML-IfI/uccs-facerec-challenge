# This file contains the source code to run the face detector on the validation and test set images
import torch
from PIL import Image
import os
from facenet_pytorch import MTCNN
from tqdm import tqdm

def detect_faces(img_files,thresholds,max_detections,logger,gpu_index=None):
  """
  This function detects faces on given img paths by using MTCNN model
  """

  # device for the inference
  device = torch.device(f"cuda:{gpu_index}") if gpu_index >= 0 else torch.device("cpu")

  # if select_largest is False, all bboxes are sorted by their detection probabilities, otherwise their size
  face_detector = MTCNN(min_face_size=40, factor=0.709, thresholds=thresholds, keep_all=True,select_largest=False,device=device)

  detections = {}

  for img_file in tqdm(img_files):
    try:
      # load image; RGB PIL Image for MTCNN
      img = Image.open(img_file)

      # get bounding boxes and confidences
      faces = face_detector.detect(img,landmarks=True)

      if faces[0] is not None:
        # they are all sorted based on their detection probability
        bboxes,qualities,landmarks = faces

      else:
        # no faces detected; it will not be saved in .csv file, because it is a empty list
        bboxes, qualities,landmarks = [], [],[]
        logger.warning("No face was found for image %s", os.path.basename(img_file))

      detections[os.path.basename(img_file)] = [(qualities[i],bboxes[i],landmarks[i]) for i in range(min(max_detections, len(qualities)))]

    except Exception as e:
      logger.error("File %s: error %s",img_file,e)

  return detections


def save_detections(detections,saving_path):
  """
  This writes all detection results to specified file.
  """

  with open(saving_path, "w") as f:

    f.write("FILE,DETECTION_SCORE,BB_X,BB_Y,BB_WIDTH,BB_HEIGHT,REYE_X,REYE_Y,LEYE_X,LEYE_Y,NOSE_X,NOSE_Y,RMOUTH_X,RMOUTH_Y,LMOUTH_X,LMOUTH_Y\n")

    for image in sorted(detections.keys()):

      for (score,bbox,lmark) in detections[image]:

        f.write("%s,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f,%3.2f\n" % (image, score,
                                                                                      bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1],
                                                                                      lmark[0][0],lmark[0][1],
                                                                                      lmark[1][0],lmark[1][1],
                                                                                      lmark[2][0],lmark[2][1],
                                                                                      lmark[3][0],lmark[3][1],
                                                                                      lmark[4][0],lmark[4][1]))
