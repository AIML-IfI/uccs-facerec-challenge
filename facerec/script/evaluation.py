# This file contains code to produce a score file and plots F-ROC and O-ROC file for the baseline
# It can be used for either all of them or just one of them (detection, score file, identification)
# But arguments that are below the interested section should be given
import logging
import yamlparser
import argparse
import os
import pickle
import numpy as np
from ..dataset import read_detections,read_ground_truth,read_recognitions
from ..evaluate import assign_detections,compute_DR_FDPI,plot_froc_curve,compute_TPIR_FPIPI,plot_oroc_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRec.UCCS")

def read_config_file():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", "-t",
        nargs = "+",
        choices = ["detection", "recognition"],
        default = ["detection", "recognition"],
        help = "Select the tasks that should be performed in this evaluation"
    )

    parent_direct = os.path.dirname(os.path.dirname(__file__))
    cfg = yamlparser.config_parser(parser=parser,default_config_files=[os.path.join(parent_direct, "configs/baseline_config.yaml")])

    if 'detection' in cfg.tasks:
        if cfg.eval.detection.files is None:
            raise ValueError("For the detection task, --eval.detection.files are required.")

        if cfg.eval.detection.labels is None:
            cfg.eval.detection.labels = cfg.eval.detection.files

        if len(cfg.eval.detection.labels) != len(cfg.eval.detection.files):
            raise ValueError("The number of labels (%d) and results (%d) differ" % (len(cfg.eval.detection.labels), len(cfg.eval.detection.files)))

    if 'recognition' in cfg.tasks:
        if cfg.eval.recognition.files is None:
            raise ValueError("For the identification task, --eval.recognition.files are required.")

        if cfg.eval.recognition.labels is None:
            cfg.eval.recognition.labels = cfg.eval.recognition.files

        if len(cfg.eval.recognition.labels) != len(cfg.eval.recognition.files):
            raise ValueError("The number of labels (%d) and results (%d) differ" % (len(cfg.eval.recognition.labels), len(cfg.eval.recognition.labels)))

    return cfg

def main():

    # get config arguments
    cfg = read_config_file()

    # load the evaluation protocol
    logger.info("Loading UCCS %s evaluation protocol",cfg.which_set)

    #read detections based on the task (detection/identification)
    if "detection" in cfg.tasks or "recognition" in cfg.tasks:
        # read the ground truth bounding boxes of the validation set
        logger.info("Reading UCCS %s ground-truth from the protocol",cfg.which_set)
        ground_truth = read_ground_truth(cfg.data_directory,cfg.which_set)

        face_numbers = sum([len(face_ids) for face_ids,_,_ in ground_truth.values()])
        image_numbers = len(ground_truth)

        if cfg.eval.exclude_gallery is not None:
            with open(cfg.eval.exclude_gallery, 'rb') as file:
                exclude = pickle.load(file)
            # update the number of faces in the set
            face_numbers -= len(exclude)

    # plot f-roc for the detection results if it is given
    if  "detection" in cfg.tasks:

        detection_results = []

        for idx,detection_file in enumerate(cfg.eval.detection.files):

            logger.info("Reading detections from %s (%s)", detection_file, cfg.eval.detection.labels[idx])
            detections = read_detections(detection_file)

            matched_detections = assign_detections(ground_truth,detections,cfg.eval.iou,exclude)

            logger.info("Computing DR and FDPI for %s (%s)", detection_file, cfg.eval.detection.labels[idx])
            DR,FDPI = compute_DR_FDPI(matched_detections,face_numbers,image_numbers,cfg.eval.detection.plot_numbers)

            detection_results.append((DR,FDPI))

        # plotting
        froc_path = cfg.eval.detection.froc
        logger.info("Plotting F-ROC curve(s) to file '%s'", froc_path)
        plot_froc_curve(detection_results,cfg.eval.detection.labels,froc_path,
                                 face_numbers,cfg.eval.linear,cfg.eval.detection.plot_numbers)

    # plot o-roc for the identification results if it is given
    if "recognition" in cfg.tasks:

        known_numbers = sum([ sum(np.array(subject_ids) > 0) for _,subject_ids,_ in ground_truth.values()])
        if exclude:
            known_numbers -= len(exclude)

        recognition_results = []

        for idx, score_file in enumerate(cfg.eval.recognition.files):
            logger.info("Reading scores from %s (%s)", score_file, cfg.eval.recognition.labels[idx])
            all_scores = read_recognitions(score_file)

            matched_detections = assign_detections(ground_truth,all_scores,cfg.eval.iou,exclude)

            logger.info("Computing TPIR and FPIPI for %s (%s)", score_file, cfg.eval.recognition.labels[idx])
            TPIR,FPIPI = compute_TPIR_FPIPI(all_scores,matched_detections,known_numbers,image_numbers,
                                                    cfg.eval.recognition.rank,cfg.eval.recognition.plot_numbers)

            recognition_results.append((TPIR,FPIPI))

        # plotting
        oroc_path = cfg.eval.recognition.oroc
        logger.info("Plotting O-ROC curve(s) to file '%s'", oroc_path)
        plot_oroc_curve(recognition_results,cfg.eval.recognition.labels,cfg.eval.recognition.rank,oroc_path,
                                 known_numbers,linear=cfg.eval.linear,
                                 plot_recognition_numbers=cfg.eval.recognition.plot_numbers)

if __name__ == "__main__":
    main()
