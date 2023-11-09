# This file contains code to produce a score file and plots F-ROC and O-ROC file for the baseline
# It can be used for either all of them or just one of them (detection, score file, identification)
# But arguments that are below the interested section should be given
import logging
import argparse
import os
import pickle
import numpy as np
from ..enrollment import average
from ..scoring import create_score_file
from ..dataset import read_detections,read_ground_truth,read_recognitions
from ..evaluate import assign_detections,compute_DR_FDPI,plot_froc_curve,compute_TPIR_FPIPI,plot_oroc_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRec.UCCS")

def command_line_options(command_line_arguments):

    parser = argparse.ArgumentParser(description="Get detection, scoring and/or identification results")

    # common options
    parser.add_argument('--task', '-t',required=True,nargs='+', help='Choose the task(s): scoring, detection, or identification')
    parser.add_argument('--data-dir', '-d', required=True, help="Select the directory where the UCCS data files are stored")
    parser.add_argument('--which-set', '-s', required=True, default="validation", choices=("validation", "test"), help="Select the protocol to use")
    parser.add_argument('--exclude-gallery','-ex',help="Select the file where gallery face IDs are stored to exclude them from the results")
    parser.add_argument('--linear-scale', '-x', action='store_true', help="If selected, plots will be in linear, otherwise semilogx")
    
    # detection task requires '--detection-files'
    parser.add_argument('--detection-files', '-df', nargs='+', help="Get the file(s) containing the UCCS face detection results")
    parser.add_argument('--iou-threshold', '-T', type=float, default=0.5, help="The overlap threshold for detected faces to be considered to be detected correctly")
    parser.add_argument('--detection-labels', '-dl', nargs='+', help="Use these labels; if not given, the filenames will be used")
    parser.add_argument('--froc-file', '-wf', default="results/UCCS-FROC.pdf", help="The file where the FROC curve will be plotted into")
    parser.add_argument('--plot-detection-numbers', '-pd', action='store_true', help="If selected, the total number of detected faces will be shown (rather than percentages)")

    # scoring task requires '--gallery-dir' and '--probe-dir' arguments
    parser.add_argument('--gallery-dir', '-g', help="Select the directory where the gallery embeddings are stored")
    parser.add_argument('--probe-dir', '-p', help="Select the directory where the probe embeddings are stored")
    parser.add_argument('--file-loc', '-ws', default="results/UCCS-%s-scoring.txt", help="Save the score file as .txt file")
    
    # identification requires '--scoring-files'
    parser.add_argument('--scoring-files', '-sf', nargs='+', help="Get the file with the UCCS face detection results")
    parser.add_argument('--rank', '-R', type=int, default=1, help="Plot DIR curves for the given rank")
    parser.add_argument('--scoring-labels', '-sl', nargs='+', help="Use these labels; if not given, the directory names will be used")
    parser.add_argument('--oroc-file', '-wo', default="results/UCCS-OROC.pdf", help="The file where the O-ROC curve will be plotted into")
    parser.add_argument('--plot-recognition-numbers', '-pr', action='store_true', help="If selected, the total number of recognized faces will be shown (rather than percentages)")
    
    args = parser.parse_args(command_line_arguments)

    if 'detection' in args.task:
        if args.detection_files is None:
            parser.error("For the detection task, --detection-files are required.")

        if args.detection_labels is None:
            args.detection_labels = args.detection_files

        if len(args.detection_labels) != len(args.detection_files):
            raise ValueError("The number of labels (%d) and results (%d) differ" % (len(args.detection_labels), len(args.detection_files)))
    
    if 'scoring' in args.task:
        if (args.gallery_dir is None or args.probe_dir is None):
            parser.error("For the scoring task, both --gallery-dir and --probe-dir are required.")
        try:
            args.file_loc = args.file_loc % args.which_set
        except:
            pass
        
    if 'identification' in args.task :
        if args.scoring_files is None:
            parser.error("For the identification task, --scoring-files are required.")

        if args.scoring_labels is None:
            args.scoring_labels = args.scoring_files
    
        if len(args.scoring_labels) != len(args.scoring_files):
            raise ValueError("The number of labels (%d) and results (%d) differ" % (len(args.scoring_labels), len(args.scoring_files)))
    
    return args

def main(command_line_arguments=None):
  
    # get command line arguments
    args = command_line_options(command_line_arguments)

    # load the evaluation protocol
    logger.info("Loading UCCS %s evaluation protocol",args.which_set)

    #read detections based on the task (detection/identification)
    if "detection" in args.task or "identification" in args.task:
        # read the ground truth bounding boxes of the validation set
        logger.info("Reading UCCS %s ground-truth from the protocol",args.which_set)
        ground_truth = read_ground_truth(args.data_dir,args.which_set)

        face_numbers = sum([ len(face_ids) for face_ids,_,_ in ground_truth.values()])
        image_numbers = len(ground_truth)

        if args.exclude_gallery is not None:
            with open(os.path.join(args.data_dir,args.exclude_gallery), 'rb') as file:
                exclude = pickle.load(file)
            # update the number of faces in the set
            face_numbers -= len(exclude)

    # plot f-roc for the detection results if it is given
    if  "detection" in args.task:

        detection_results = []

        for idx,detection_file in enumerate(args.detection_files):

            logger.info("Reading detections from %s (%s)", detection_file, args.detection_labels[idx])
            detections = read_detections(os.path.join(args.data_dir,detection_file))

            matched_detections = assign_detections(ground_truth,detections,args.iou_threshold,exclude)

            logger.info("Computing DR and FDPI for %s (%s)", detection_file, args.detection_labels[idx])
            DR,FDPI = compute_DR_FDPI(matched_detections,face_numbers,image_numbers,args.plot_detection_numbers)

            detection_results.append((DR,FDPI))

        # plotting
        logger.info("Plotting F-ROC curve(s) to file '%s'", args.froc_file)
        froc_path = os.path.join(args.data_dir,args.froc_file)
        plot_froc_curve(detection_results,args.detection_labels,froc_path,
                                 face_numbers,args.linear_scale,args.plot_detection_numbers)
        
    # create score file if it is given
    if "scoring" in args.task:
        logger.info("Loading UCCS %s scoring protocol",args.which_set)
        # get gallery enrollment
        logger.info("Getting UCCS gallery enrollment (average)")
        gallery_embedd_path = os.path.join(args.data_dir,args.gallery_dir)
        subject_ids,gallery_enroll = average(gallery_embedd_path)

        subject_ids = ["S_"+i for i in subject_ids]

        # compute scores between enrollment and probe and write them into a file
        probe_path = os.path.join(args.data_dir,args.probe_dir)
        scoring_path = os.path.join(args.data_dir,args.file_loc)
        logger.info("Computing scores and writing them into %s",args.file_loc)
        _ = create_score_file((subject_ids,gallery_enroll),probe_path,scoring_path)

    # plot o-roc for the identification results if it is given
    if "identification" in args.task:
        
        known_numbers = sum([ sum(np.array(subject_ids) > 0) for _,subject_ids,_ in ground_truth.values()])
        if exclude:
            known_numbers -= len(exclude)

        recognition_results = []

        for idx, score_file in enumerate(args.scoring_files):
            logger.info("Reading scores from %s (%s)", score_file, args.scoring_labels[idx])
            scoring_path = os.path.join(args.data_dir,score_file) 
            all_scores = read_recognitions(scoring_path)

            matched_detections = assign_detections(ground_truth,all_scores,args.iou_threshold,exclude)

            logger.info("Computing TPIR and FPIPI for %s (%s)", score_file, args.scoring_labels[idx])
            TPIR,FPIPI = compute_TPIR_FPIPI(all_scores,matched_detections,known_numbers,image_numbers,
                                                    args.rank,args.plot_recognition_numbers)

            recognition_results.append((TPIR,FPIPI))

        # plotting
        logger.info("Plotting O-ROC curve(s) to file '%s'", args.oroc_file)
        oroc_path = os.path.join(args.data_dir,args.oroc_file)
        plot_oroc_curve(recognition_results,args.scoring_labels,args.rank,oroc_path,
                                 known_numbers,linear=args.linear_scale,
                                 plot_recognition_numbers=args.plot_recognition_numbers)
        
if __name__ == "__main__":
    main()
