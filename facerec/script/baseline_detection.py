# This file contains a script to run the MTCNN (baseline face detector) on the validation and test set images,
# and writes them into a file
import yamlparser
import logging
import os
import multiprocessing
from ..dataset import read_ground_truth
from ..face_detection import detect_faces,save_detections
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UCCS.FaceRec")

def read_configuration_file():

  cfg = yamlparser.config_parser()

  if not os.path.exists(cfg.result_directory):
    os.mkdir(cfg.result_directory)

  return cfg


def main():

  # get command line arguments
  cfg = read_configuration_file()

  # load detection protocol
  logger.info("Loading UCCS %s detection protocol", cfg.which_set)
  data = read_ground_truth(cfg.data_directory,cfg.which_set)
  img_names = sorted(data.keys())

  # get the paths of images
  image_directory = cfg.format(cfg.image_directory)
  img_files = [os.path.join(image_directory,img_name) for img_name in img_names]

  if cfg.detection.parallel is None or cfg.gpu:
    logger.info("Detecting faces in %d images sequentially",len(img_names))
    detections = detect_faces(img_files,cfg.detection.thresholds,cfg.detection.max_detections,logger,cfg.gpu)

  else:
    # parallelization; split data into chunks
    logger.info("Detecting faces in %d images using %d parallel processes", len(img_names), cfg.detection.parallel)

    pool = multiprocessing.Pool(cfg.detection.parallel)

    partial_detect_faces = partial(detect_faces,thresholds=cfg.detection.thresholds,max_detections=cfg.detection.max_detections,logger=logger)
    # Split image names into chunks for parallel processing
    chunks = [([d for i, d in enumerate(img_files) if i % cfg.detection.parallel == p]) for p in range(cfg.detection.parallel)]

    # Perform parallel processing
    results = pool.map(partial_detect_faces, chunks)

    # Combine the results from all processes
    detections = {}
    for result in results:
      detections.update(result)

    pool.close()
    pool.join()

  result_file = cfg.format(cfg.detection.results)
  logger.info("Writing detections to file %s", result_file)
  save_detections(detections, result_file)

if __name__ == "__main__":
    main()
