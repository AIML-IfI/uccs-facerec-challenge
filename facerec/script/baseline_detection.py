# This file contains a script to run the MTCNN (baseline face detector) on the validation and test set images, 
# and writes them into a file
import argparse
import logging
from PIL import Image
import os
import multiprocessing
from facenet_pytorch import MTCNN
import torch
from ..dataset import read_ground_truth
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UCCS.FaceRec")

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description="Detect faces in a set of UCCS")

  parser.add_argument('--data-directory', '-d', required=True, help = "Select the directory, where the UCCS files are stored")
  parser.add_argument('--image-directory','-i',required=True, help = "Select the directory, where the images are stored")
  parser.add_argument('--result-file', '-w', default = "results/UCCS-detection-baseline-%s.txt", help = "Select the file to write the scores into")
  parser.add_argument('--which-set', '-s', default = "validation", choices = ("validation", "test"), help = "Select the protocol to use")

  # the arguments for the baseline detector's (MTCNN) parameters
  parser.add_argument('--thresholds', '-T',nargs='+',required=True,type=float,default=[0.2,0.2,0.2], help = "Limits detections to those that have a prediction value higher than --threshold")
  parser.add_argument('--maximum-detections', '-M', type=int, default=50, help = "Specify, how many detections per image should be stored")

  parser.add_argument('--gpu','-g', type=int, help='GPU indice to run the detector on it (e.g., --gpu 0 ), otherwise cpu will be used')
  parser.add_argument('--parallel', '-P', type=int, help = "If given, images will be processed with the given number of parallel processes")

  args = parser.parse_args(command_line_arguments)

  try:
    args.result_file = args.result_file % args.which_set
  except:
    pass

  return args

def _detect(arguments):
  
  img_files,args = arguments

  # device for the inference
  device = torch.device(f"cuda:{args.gpu}") if isinstance(args.gpu,int) else torch.device("cpu")

  # if select_largest is False, all bboxes are sorted by their detection probabilities, otherwise their size
  face_detector = MTCNN(min_face_size=40, factor=0.709, thresholds=args.thresholds, keep_all=True,select_largest=False,device=device)

  detections = {}

  for image_file in tqdm(img_files):
    try:
      # load image; RGB PIL Image for MTCNN
      img = Image.open(os.path.join(args.data_directory,args.image_directory,image_file))

      # get bounding boxes and confidences
      faces = face_detector.detect(img,landmarks=True)

      if faces[0] is not None:
        # they are all sorted based on their detection probability
        bboxes,qualities,landmarks = faces

      else:
        # no faces detected; it will not be saved in .csv file, because it is a empty list
        bboxes, qualities,landmarks = [], [],[]
        logger.warning("No face was found for image %s", image_file)

      detections[image_file] = [(qualities[i],bboxes[i],landmarks[i]) for i in range(min(args.maximum_detections, len(qualities)))]
      
    except Exception as e:
      logger.error("File %s: error %s",image_file,e)
  
  return detections

def main(command_line_arguments=None):
  
  # get command line arguments
  args = command_line_options(command_line_arguments)

  # load detection protocol
  logger.info("Loading UCCS %s detection protocol", args.which_set)
  data = read_ground_truth(args.data_directory,args.which_set)
  img_names = sorted(data.keys())
  
  if args.parallel is None or args.gpu:
    logger.info("Detecting faces in %d images sequentially",len(img_names))
    detections = _detect((img_names,args))

  else:
    # parallelization; split data into chunks
    logger.info("Detecting faces in %d images using %d parallel processes", len(img_names), args.parallel)

    pool = multiprocessing.Pool(args.parallel)

    # Split image names into chunks for parallel processing
    chunks = [([d for i, d in enumerate(img_names) if i % args.parallel == p], ) + (args,) for p in range(args.parallel)]

    # Perform parallel processing
    results = pool.map(_detect, chunks)

    # Combine the results from all processes
    detections = {}
    for result in results:
      detections.update(result)

    pool.close()
    pool.join()
  
  result_path = os.path.join(args.data_directory,args.result_file)
  logger.info("Writing detections to file %s", args.result_file)

  with open(result_path, "w") as f:

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
if __name__ == "__main__":
    main()

