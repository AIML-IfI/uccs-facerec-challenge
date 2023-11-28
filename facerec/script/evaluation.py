# This file contains code to produce a score file and plots F-ROC and O-ROC file for the baseline
# It can be used for either all of them or just one of them (detection, score file, identification)
# But arguments that are below the interested section should be given
import logging
import yamlparser
import os
import pickle
import numpy as np
from ..enrollment import average
from ..scoring import create_score_file
from ..dataset import read_detections,read_ground_truth,read_recognitions
from ..evaluate import assign_detections,compute_DR_FDPI,plot_froc_curve,compute_TPIR_FPIPI,plot_oroc_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRec.UCCS")

def read_config_file():
    args = yamlparser.config_parser()
    args.data_directory = os.path.abspath(args.data_directory)

    if 'detection' in args.eval.task:
        if args.eval.detection.files is None:
            raise ValueError("For the detection task, --eval.detection.files are required.")

        if args.eval.detection.labels is None:
            args.eval.detection.labels = args.eval.detection.files

        if len(args.eval.detection.labels) != len(args.eval.detection.files):
            raise ValueError("The number of labels (%d) and results (%d) differ" % (len(args.eval.detection.labels), len(args.eval.detection.files)))
    
    if 'scoring' in args.eval.task:
        if (args.eval.scoring.gallery is None or args.eval.scoring.probe is None):
            raise ValueError("For the scoring task, both --eval.scoring.gallery and --eval.scoring.probe are required.")
        try:
            args.eval.scoring.results = args.eval.scoring.results % args.which_set
        except:
            pass
        
    if 'recognition' in args.eval.task :
        if args.eval.recognition.files is None:
            raise ValueError("For the identification task, --eval.recognition.files are required.")

        if args.eval.recognition.labels is None:
            args.eval.recognition.labels = args.eval.recognition.files
    
        if len(args.eval.recognition.labels) != len(args.eval.recognition.files):
            raise ValueError("The number of labels (%d) and results (%d) differ" % (len(args.eval.recognition.labels), len(args.eval.recognition.labels)))
    
    return args

def main():
  
    # get command line arguments
    args = read_config_file()

    # load the evaluation protocol
    logger.info("Loading UCCS %s evaluation protocol",args.which_set)

    #read detections based on the task (detection/identification)
    if "detection" in args.eval.task or "identification" in args.eval.task:
        # read the ground truth bounding boxes of the validation set
        logger.info("Reading UCCS %s ground-truth from the protocol",args.which_set)
        ground_truth = read_ground_truth(args.data_directory,args.which_set)

        face_numbers = sum([ len(face_ids) for face_ids,_,_ in ground_truth.values()])
        image_numbers = len(ground_truth)

        if args.eval.exclude_gallery is not None:
            with open(os.path.join(args.data_directory,args.eval.exclude_gallery), 'rb') as file:
                exclude = pickle.load(file)
            # update the number of faces in the set
            face_numbers -= len(exclude)

    # plot f-roc for the detection results if it is given
    if  "detection" in args.eval.task:

        detection_results = []

        for idx,detection_file in enumerate(args.eval.detection.files):

            logger.info("Reading detections from %s (%s)", detection_file, args.eval.detection.labels[idx])
            detections = read_detections(os.path.join(args.data_directory,detection_file))

            matched_detections = assign_detections(ground_truth,detections,args.eval.iou,exclude)

            logger.info("Computing DR and FDPI for %s (%s)", detection_file, args.eval.detection.labels[idx])
            DR,FDPI = compute_DR_FDPI(matched_detections,face_numbers,image_numbers,args.eval.detection.plot_numbers)

            detection_results.append((DR,FDPI))

        # plotting
        logger.info("Plotting F-ROC curve(s) to file '%s'", args.eval.detection.froc)
        froc_path = os.path.join(args.data_directory,args.eval.detection.froc)
        plot_froc_curve(detection_results,args.eval.detection.labels,froc_path,
                                 face_numbers,args.eval.linear,args.eval.detection.plot_numbers)
        
    # create score file if it is given
    if "scoring" in args.eval.task:
        logger.info("Loading UCCS %s scoring protocol",args.which_set)
        # get gallery enrollment
        logger.info("Getting UCCS gallery enrollment (average)")
        gallery_embedd_path = os.path.join(args.data_directory,args.eval.scoring.gallery)
        subject_ids,gallery_enroll = average(gallery_embedd_path)

        subject_ids = ["S_"+i for i in subject_ids]

        # compute scores between enrollment and probe and write them into a file
        probe_path = os.path.join(args.data_directory,args.eval.scoring.probe)
        scoring_path = os.path.join(args.data_directory,args.eval.scoring.results)
        logger.info("Computing scores and writing them into %s",args.eval.scoring.results)
        _ = create_score_file((subject_ids,gallery_enroll),probe_path,scoring_path)

    # plot o-roc for the identification results if it is given
    if "recognition" in args.eval.task:
        
        known_numbers = sum([ sum(np.array(subject_ids) > 0) for _,subject_ids,_ in ground_truth.values()])
        if exclude:
            known_numbers -= len(exclude)

        recognition_results = []

        for idx, score_file in enumerate(args.eval.recognition.files):
            logger.info("Reading scores from %s (%s)", score_file, args.eval.recognition.files[idx])
            scoring_path = os.path.join(args.data_directory,score_file) 
            all_scores = read_recognitions(scoring_path)

            matched_detections = assign_detections(ground_truth,all_scores,args.eval.iou,exclude)

            logger.info("Computing TPIR and FPIPI for %s (%s)", score_file, args.eval.recognition.labels[idx])
            TPIR,FPIPI = compute_TPIR_FPIPI(all_scores,matched_detections,known_numbers,image_numbers,
                                                    args.eval.recognition.rank,args.eval.recognition.plot_numbers)

            recognition_results.append((TPIR,FPIPI))

        # plotting
        logger.info("Plotting O-ROC curve(s) to file '%s'", args.eval.recognition.oroc)
        oroc_path = os.path.join(args.data_directory,args.eval.recognition.oroc)
        plot_oroc_curve(recognition_results,args.eval.recognition.labels,args.eval.recognition.rank,oroc_path,
                                 known_numbers,linear=args.eval.linear,
                                 plot_recognition_numbers=args.eval.recognition.plot_numbers)
        
if __name__ == "__main__":
    main()
