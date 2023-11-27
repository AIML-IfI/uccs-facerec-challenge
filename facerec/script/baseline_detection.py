# This file contains a script to run the MTCNN (baseline face detector) on the validation and test set images, 
# and writes them into a file
import yamlparser
import logging
import os
import multiprocessing
from ..dataset import read_ground_truth
from ..face_detection import detect_faces,save_detections

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UCCS.FaceRec")

def read_configuration_file():

  args = yamlparser.config_parser()
  args.data_directory = os.path.abspath(args.data_directory)

  try:
    args.baseline_detection.results = args.baseline_detection.results % args.which_set
  except:
    pass
  
  args.baseline_detection.results = os.path.join(args.data_directory,args.baseline_detection.results)

  if os.path.exists(os.path.dirname(args.baseline_detection.results)):
    os.mkdir(os.path.dirname(args.baseline_detection.results))

  return args

def main():
  
  # get command line arguments
  args = read_configuration_file()

  # load detection protocol
  logger.info("Loading UCCS %s detection protocol", args.which_set)
  data = read_ground_truth(args.data_directory,args.which_set)
  img_names = sorted(data.keys())

  # get the paths of images
  img_files = [os.path.join(args.data_directory,args.image_directory,img_name) for img_name in img_names]
  # if not given, it will be run on cpu
  gpu_index = args.gpu[0] if args.gpu else None   

  if args.baseline_detection.parallel is None or args.gpu:
    logger.info("Detecting faces in %d images sequentially",len(img_names))
    detections = detect_faces(img_files,args.baseline_detection.thresholds,args.baseline_detection.max_detections,logger,gpu_index)
    
  else:
    # parallelization; split data into chunks
    logger.info("Detecting faces in %d images using %d parallel processes", len(img_names), args.baseline_detection.parallel)

    pool = multiprocessing.Pool(args.baseline_detection.parallel)

    arguments = (args.baseline_detection.thresholds,args.baseline_detection.max_detections,logger,gpu_index)
    # Split image names into chunks for parallel processing
    chunks = [([d for i, d in enumerate(img_files) if i % args.baseline_detection.parallel == p], ) + arguments for p in range(args.baseline_detection.parallel)]

    # Perform parallel processing
    results = pool.map(detect_faces, chunks)

    # Combine the results from all processes
    detections = {}
    for result in results:
      detections.update(result)

    pool.close()
    pool.join()
  
  logger.info("Writing detections to file %s", )
  save_detections(detections,args.baseline_detection.results)

if __name__ == "__main__":
    main()

