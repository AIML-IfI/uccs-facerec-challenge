# This file contains functionality to process scores in order to create O-ROC plots
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import box_iou
import torch
from termcolor import cprint

def get_iou_pairwise(ground_truth_bboxes,detection_bboxes):
    """
    It computes the standard iou scores between ground truth and detection bboxes
    """
    # convert it to tensor and x1,y1,x2,y2 format 
    gr = torch.Tensor(ground_truth_bboxes)
    gr[:,2] += gr[:,0]
    gr[:,3] += gr[:,1]
    dt = torch.Tensor(detection_bboxes)
    dt[:,2] += dt[:,0]
    dt[:,3] += dt[:,1]
    
    assert gr.shape == (len(gr),4) and dt.shape==(len(dt),4)

    # get the pairwise of iou scores between ground truth and detections
    iou_pairwise = box_iou(gr,dt)

    return iou_pairwise.numpy()

def assign_detections(ground_truth,proposals,overlap_threshold=0.5,exclude=None):
    """
    It matches bounding boxes(proposals) with ground truth (if possible) or assigns it as a misdetection.
    Propasals can come from the detection files or scoring files

    A detection can match with only one unique ground truth in this matching procedure.

    Note that gallery faces will be excluded from the results when their face_ids are given in the the parameter of 'exclude'. 
    """
    assigned_detections = {}
    # store all detection scores [scores..],matched_overlaps -> [(indice_proposal,subject_id),..], misdetections-> [indices..]
    for img, (face_ids, subject_ids, ground_truth_bboxes) in ground_truth.items():
        
        if img not in proposals:
            continue

        detection_scores,prop_bboxes,_ = proposals[img]
        prop_bboxes = np.array(prop_bboxes)

        iou_pairwise = get_iou_pairwise(ground_truth_bboxes,prop_bboxes)

        # initialize arrays to keep track of matches
        matched_indices = []  # store pairs of matched ground truth and proposal indices
        unmatched_detection_indices = list(range(len(prop_bboxes)))  # initially, all proposals are unmatched

        while True:
            # find the maximum IoU in the current IoU matrix
            max_iou = np.max(iou_pairwise)
            
            if max_iou < overlap_threshold:
                break  # exit if no IoU above the threshold is found

            # find the indices of the maximum IoU
            gt_index, proposal_index = np.unravel_index(np.argmax(iou_pairwise), iou_pairwise.shape)

            # exclude gallery faces if it is given
            if exclude and face_ids[gt_index] in exclude:
                # if it is matched, dont add the detection to false positives as well as true positives
                unmatched_detection_indices.remove(proposal_index)
                iou_pairwise[gt_index, :] = 0
                iou_pairwise[:, proposal_index] = 0
                continue

            # add the matched pair to the list -> proposal index : subject_id of the ground truth
            matched_indices.append((proposal_index,subject_ids[gt_index]))
            unmatched_detection_indices.remove(proposal_index)

            # set the IoU values for the corresponding ground truth and proposals to 0
            iou_pairwise[gt_index, :] = 0
            iou_pairwise[:, proposal_index] = 0
        
        # add the image results to the dictionary
        assigned_detections[img] = (detection_scores,matched_indices,unmatched_detection_indices)

    return assigned_detections

def compute_DR_FDPI(all_matched_detections,face_numbers,image_numbers,plot_detection_numbers=False):
    """
    It computes Detection Rate (true positive rate) and False Detection Per Image (FDPI) for plotting.
    """

    positives = []
    negatives = []

    for _, (detection_scores, matched_indices, misdetections) in all_matched_detections.items():

        # detections matched with a ground-truth is assigned as a positive score 
        pos = [detection_scores[detection_ind] for detection_ind,_ in matched_indices]
        positives.extend(pos)

        # detections that are not matched with any of ground truth 
        neg = [detection_scores[detection_ind] for detection_ind in misdetections]
        negatives.extend(neg)

    cprint(f"In total: {face_numbers} faces, {len(positives)} detected faces and {len(negatives)} false detections",
           'green', attrs=['bold', 'underline'])
    positives = np.array(positives)
    negatives = np.array(negatives)

    # thresholds
    # thresholds = np.linspace(min(negatives), max(negatives), 100)
    thresholds = np.unique(negatives)

    # detection rate and false detection per image
    DR = []
    FDPI = []

    for thr in thresholds:

        dr = np.sum(positives >= thr)

        if not plot_detection_numbers:
            # get the rate
            dr /= face_numbers

        fdpi = np.sum(negatives >= thr) / image_numbers # the number of images in the probe

        DR.append(dr)
        FDPI.append(fdpi)

    return DR,FDPI

def compute_TPIR_FPIPI(all_scores,all_matched_detections,known_numbers,image_numbers,rank=1,plot_recognition_numbers=False):
    """
    It computes True Positive Identification Rate and False Postive Identification Per Image (FPIPI) for plotting.
    TPIR = True Positives / the number of known faces in the probe
    FPIPI = False Positives / the number of probe images
    """

    # get postive and negative scores
    positives = []
    negatives = []
    
    for img, (_, matched_detections, misdetections) in all_matched_detections.items():
            
        # get the score for each detection in the image
        _, _, scores =  all_scores[img]
        similarity_score = np.array(scores)
        
        # get score indices based on the rank
        sort = np.argsort(similarity_score[detection_indice])[::-1][:rank]
        
        #count true positive identification: there are two factors to be counted as true positive identification
        #1. First, the detection bbox must be matched with ground truth
        #2. Second, the similarity score of this matching should match with the subject id based on the threshold
        for detection_indice,sub_id in matched_detections:
            
            if sub_id > 0:
                best_ids = sort + 1  # because scores are sorted based on idendity number starting 1

                if sub_id in best_ids:
                    ind_id = np.where(best_ids==sub_id)[0][0]
                    pos_score = similarity_score[detection_indice][sort][ind_id]
                    positives.append(pos_score)

            # if the detection is unknown, get the max score as a negative
            else:
                negatives.append(similarity_score[detection_indice][sort[0]])
        
        # add also background detections to the negative scores
        negatives.extend([ np.max(similarity_score[mis_ind]) for mis_ind in misdetections])

    positives = np.array(positives)
    negatives = np.array(negatives)

    # thresholds
    # thresholds = np.linspace(min(negatives), max(negatives), 100)
    thresholds = np.unique(negatives)

    # tpir and fpipi for each thr
    TPIR = []
    FPIPI = []

    for thr in thresholds:

        tp = np.sum(positives >= thr)

        # normalize by the known counter
        if not plot_recognition_numbers:
            tp /= known_numbers

        fp = np.sum(negatives>=thr) / image_numbers # the number of images in the probe

        TPIR.append(tp)
        FPIPI.append(fp)

    return TPIR,FPIPI

def plot_oroc_curve(results,labels,rank,saving_path,known_numbers,linear=False,plot_recognition_numbers=False):
    """
    This function plots O-ROC curve (TPIR-FPIPI) based on given results

    results: a list of [TPIR,FPIPI]s 
    """
    # create the figure
    figure = plt.figure(figsize=(7,4.5))
    # set the plotter function based on the linear flag
    plotter = plt.semilogx if not linear else plt.plot
    
    max_fpi = 0
    min_fpi = np.inf
    for ix, label in enumerate(labels):
        plotter(results[ix][1], results[ix][0], label=label)
        
        max_fpi = max(max_fpi,results[ix][1][0]) # the first one is the biggest because its thr is the smallest
        min_fpi = min(min_fpi,results[ix][1][-1])

    plt.grid(True, color=(0.6,0.6,0.6))
    plt.title("Rank %d O-ROC curve" % rank)

    plt.legend(loc=2 if not linear else 4, prop={'size':14})
    plt.xlabel('FPI Per Image')

    if not linear:
        plt.xlim([min_fpi,max_fpi])
    else:
        plt.xlim([0,max_fpi])

    if plot_recognition_numbers:
        plt.ylim((0, known_numbers))
        plt.ylabel('TPI')
    else:
        plt.ylim((0, 1))
        plt.ylabel('TPIR Per Face')
    
    plt.tight_layout()
    plt.savefig(saving_path)

def plot_froc_curve(results,labels,saving_path,face_numbers,linear=False,plot_detection_numbers=False):
    """
    This function plots F-ROC curve (DR-FDPI) based on given results

    results: a list of [DR,FDPI]s 
    """
    # create the figure
    figure = plt.figure(figsize=(7,4.5))
    # set the plotter function based on the linear flag
    plotter = plt.semilogx if not linear else plt.plot

    max_fd = 0
    min_fd = np.inf
    for ix, label in enumerate(labels):
        plotter(results[ix][1], results[ix][0], label=label)

        max_fd = max(max_fd,results[ix][1][0]) # the first index is the biggest because its thr was the smallest
        min_fd = min(min_fd,results[ix][1][-1])

    plt.grid(True, color=(0.6,0.6,0.6))

    plt.legend(loc=2 if not linear else 4, prop={'size':14})
    plt.xlabel('False Detection Per Image')

    if not linear:
        plt.xlim([min_fd,max_fd])
    else:
        plt.xlim([0,max_fd])

    if plot_detection_numbers:
        plt.ylim((0, face_numbers))
        plt.ylabel('Number of Detections')
    else:
        plt.ylim((0, 1))
        plt.ylabel('Detection Rate')
    
    plt.tight_layout()
    plt.savefig(saving_path)
