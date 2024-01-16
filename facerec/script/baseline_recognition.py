# This file contains a script to take the detected faces and process them with a face recognition algorithm
# It can extract features from both gallery and given set
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

    parent_direct = os.path.dirname(os.path.dirname(__file__))
    cfg = yamlparser.config_parser(default_config_files=[os.path.join(parent_direct, "configs/baseline_config.yaml")])

    if not cfg.recognition.detection_file:
        raise ValueError (f"For the {cfg.which_set} set, --recognition.detection_file is required.")

    if not os.path.exists(cfg.result_directory):
        os.mkdir(cfg.result_directory)

    return cfg

def main():

    # get command line arguments
    cfg = read_config_file()

    # load extraction protocol
    logger.info("Loading {}%s extraction protocol".format("gallery and " if cfg.extract_gallery else ""), cfg.which_set)

    # download MagFace repo and its model weights if it wasn't downloaded before
    logger.info("Downloading/Activating MagFace and its model weights")
    download_MagFace(cfg,logger)

    # get the model in eval mode
    logger.info("Loading the baseline model")
    model,device = build_model(cfg)

    # sets that will be extracted
    sets = ["gallery",cfg.which_set] if cfg.extract_gallery else [cfg.which_set] 

    for set_name in sets:
        # update the config.which_set
        cfg.which_set = set_name
        
        # read the detections for gallery,valid or test
        if cfg.which_set =="gallery":
            data = read_ground_truth(cfg.data_directory,"gallery")
        else:
            data = read_detections(cfg.recognition.detection_file)

        landmarks = [l for _,_,l in data.values()]
        img_files = list(data.keys())

        # create the path of images, it differs based on the protocol because the gallery has sub-directories
        image_directory = cfg.image_directory
        img_paths = [os.path.join(image_directory,file.split("_")[0],file) for file in img_files] if cfg.which_set == "gallery"  else [
        os.path.join(image_directory,file) for file in img_files]

        # get the data loader
        logger.info("Creating the %s dataloader", cfg.which_set)
        inf_loader,batch_size_perImg = inference_dataloader((img_paths,None,landmarks),
                                      cfg.which_set,cfg.batch_size_perImg,cfg.recognition.workers)

        # create result dir based on the set
        result_dir = cfg.recognition.result_dir
        os.makedirs(result_dir, exist_ok=True)

        logger.info(f"Starting the {set_name} inference process")
        with torch.no_grad():

            for input, img_names in tqdm(inf_loader):

                # when batch size = 1, to avoid extra dimension increased by dataloader
                input = input[0] if batch_size_perImg == 1 else input

                # compute features
                input = input.to(device,dtype=torch.float)
                embedding_feat = model(input)

                _feat = embedding_feat.data.cpu().numpy()

                # save all information (detection scores,bboxes,landmarks,embeddings) of that image/identity
                _ = save_features_information(data,img_names,cfg.which_set,_feat,result_dir,batch_size_perImg)

if __name__ == "__main__":
    main()
