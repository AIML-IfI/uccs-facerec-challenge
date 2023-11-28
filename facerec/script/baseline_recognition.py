# This file contains a script to take the detected faces and process them with a face recognition algorithm
import logging
import yamlparser
import os
import torch
from ..dataset import read_detections,read_ground_truth
from ..feature_extraction import download_MagFace,ImgData,inference_dataloader,build_model,save_features_information
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRec.UCCS")

def read_config_file():

    args = yamlparser.config_parser()
    args.data_directory = os.path.abspath(args.data_directory)

    if args.which_set != "gallery" and not args.baseline_recognition.detection_file:
        raise ValueError ("For the validation set, --baseline_recognition.detection_file is required.")

    try:
        args.baseline_recognition.result_dir = args.baseline_recognition.result_dir % args.which_set
        args.image_directory = args.image_directory % args.which_set
    except:
        pass

    args.baseline_recognition.result_dir = os.path.join(args.data_directory,args.baseline_recognition.result_dir)
    if not os.path.exists(args.baseline_recognition.result_dir):
        os.mkdir(args.baseline_recognition.result_dir)

    return args

def main():

    # get command line arguments
    args = read_config_file()

    # load extraction protocol
    logger.info("Loading UCCS %s extraction protocol", args.which_set)

    # read the detections for gallery,valid or test
    if args.which_set =="gallery":
        data = read_ground_truth(args.data_directory,"gallery")
    else:
        data = read_detections(os.path.join(args.data_directory,args.baseline_recognition.detection_file))
        
    landmarks = [l for _,_,l in data.values()]
    img_files = list(data.keys())

    # create the path of images, it differs based on the protocol because the gallery has sub-directories
    img_paths = [os.path.join(args.data_directory,args.image_directory,file.split("_")[0],file) for file in img_files] if args.which_set == "gallery"  else [
        os.path.join(args.data_directory,args.image_directory,file) for file in img_files]
    
    # download MagFace repo and its model weights if it wasnt downloaded before
    logger.info("Downloading/Activating MagFace and its model weights")
    download_MagFace(args,logger)

    # get the data loader
    logger.info("Creating the %s dataloader", args.which_set)
    inf_loader = inference_dataloader((img_paths,None,landmarks),
                                      args.which_set,args.batch_size_perImg,args.workers)

    # get the model in eval mode
    logger.info("Loading the baseline model")
    model,device = build_model(args)

    logger.info("Starting the inference process")
    with torch.no_grad():

        for _, (input, img_names) in tqdm(enumerate(inf_loader)):
            
            # when batch size = 1, to avoid extra dimension increased by dataloader
            input = input[0] if args.batch_size_perImg == 1 else input
            
            # compute features
            input = input.to(device,dtype=torch.float)
            embedding_feat = model(input)

            _feat = embedding_feat.data.cpu().numpy()

            # save all information (detection scores,bboxes,landmarks,embeddings) of that image/identity
            _ = save_features_information(data,img_names,args.which_set,_feat,args.baseline_recognition.result_dir,args.batch_size_perImg)

if __name__ == "__main__":
    main()
