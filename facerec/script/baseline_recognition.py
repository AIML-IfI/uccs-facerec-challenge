# This file contains a script to take the detected faces and process them with a face recognition algorithm
import sys
import logging
import argparse
import os
import numpy as np
import torch
from torchvision import transforms
import cv2
from skimage import transform
from ..dataset import read_detections,read_ground_truth
from ..feature_extraction import ImgData,save_features_information
from tqdm import tqdm
import subprocess
import gdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRec.UCCS")

def command_line_options(command_line_arguments):
    # parse the args
    parser = argparse.ArgumentParser(description='Feature extraction by MagFace')

    parser.add_argument('--data-directory', '-d', required=True, help = "Select the directory, where the UCCS files are stored")
    parser.add_argument('--image-directory','-i',required=True,help="Select the directory, where the images are stored")
    parser.add_argument('--detection-file','-df', type=str,help=' The .csv file containing the file names and landmarks that need to be extracted')
    parser.add_argument('--which-set', '-s', default = "validation", choices = ("gallery", "validation", "test"), help = "Select the protocol to use")
    parser.add_argument('--feat_dir', '-w',type=str, help='Which directory will be used for storing embeddings')

    # model arguments (MagFace)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size-perImg', default=1, type=int, metavar='N',
                        help='For validation, it should be 1 because of the different number of faces in each image'
                             'For gallery, it is multiples of 10')
    parser.add_argument('--embedding-size', default=512, type=int,
                        help='The embedding feature size')
    parser.add_argument('--gpu','-g',nargs="+", type=int, help='GPU indices to use (e.g., --gpu 0 1 ) if it will be run on gpus')

    args = parser.parse_args(command_line_arguments)

    if args.which_set != "gallery" and not args.detection_file:
        parser.error("For the validation set, --detection-file is required.")
        
    return args

def download_MagFace(args):

    magFace_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)),"MagFace")
    weights_path = os.path.join(magFace_directory,"magface_epoch_00025.pth")
    
    # downloading magface repo
    if not os.path.exists(magFace_directory):
        os.mkdir(magFace_directory)

        # the baseline requires MagFace model
        repository_url = "https://github.com/IrvingMeng/MagFace.git" 
    
        # download the default model weights from google drive (backbone--iresnet100)
        model_weights_url = "https://drive.google.com/uc?id=1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H"
        
        # construct the Git clone command
        git_clone_command = ["git", "clone", repository_url, magFace_directory]

        # run the Git clone command
        try:
            subprocess.run(git_clone_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error("Exception on process, rc= %s output= %s", e.returncode, e.output)
            os.rmdir(magFace_directory)
            sys.exit(1)
    
    # downloading magface iresnet100 model weights
    if not os.path.exists(weights_path):
        # use gdown to download the file
        try:
            gdown.download(model_weights_url, weights_path, quiet=False)
        except Exception as e:
            logger.error(f"Error: Failed to download model weights from Google Drive - {e}")
            sys.exit(1)

    # append MagFace module 
    sys.path.append(magFace_directory)

    # update the arguments for model_path and backbone architecture
    args.arch = "iresnet100"
    args.resume = weights_path

class face_normalized(ImgData):

    def __init__(self, data,which_set=None,image_size=112, align=True, transform=None):
        super().__init__(data,which_set,image_size, align, transform)

    def __getitem__(self, index):

        # get image,bboxes and landmarks
        img_path = self.img_paths[index]

        points = self.points[index]

        if self.align:
            assert points.shape == (points.shape[0],5,2) # facial landmarks
        else:
            assert points.shape == (points.shape[0],4) # bboxes

        img_name = os.path.basename(img_path)

        if not os.path.isfile(img_path):
            raise Exception('{} does not exist'.format(img_path))
        
        # read img
        img = cv2.imread(img_path)

        if img is None:
            raise Exception('{} is empty'.format(img_path))

        # crop and align faces in the image for MagFace
        faces = []
        for point in points:

            # align face
            if self.align:
                M, _ = self.estimate_norm(point)
                face = cv2.warpAffine(img, M, (self.image_size, self.image_size), borderValue=0.0)
            
            # otherwise, just crop the face
            else:
                # convert it to x1,y1,x2,y2 format
                point[:,2] += point[:,0]
                point[:,3] += point[:,1]

                # crop the face based on th bbox : x1,y1,x2,y2
                face = img[int(point[1]):int(point[3]), int(point[0]):int(point[2])]
                face = cv2.resize(face,((self.image_size, self.image_size)))
            
            assert face.shape == (self.image_size, self.image_size,3)
        
            face = self.transform(face)

            faces.append(face)
        
        if self.which_set == "gallery":
            return faces[0],img_name

        # stack all faces in the image for the extraction
        faces = torch.stack(faces)

        return faces,img_name
    
    def estimate_norm(self,lmk):
        # gets the facial landmark (reye,leye,nose,rmouth,lmouth)
        assert lmk.shape == (5, 2)
        tform = transform.SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = float('inf')
        if self.mode == 'arcface':
            if self.image_size == 112:
                src = self.arcface_src
            else:
                src = float(self.image_size) / 112 * self.arcface_src

        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))

            if error < min_error:
                min_error = error
                min_M = M
                min_index = i

        return min_M, min_index

def inference_loader(data,args):
    """
    It builds the dataloader to apply preprocesssing and get the input ready for the model.
    """
    # for preprocessing transformations
    trans = transforms.Compose([
        transforms.ToTensor(),         
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.]),
    ])

    if args.which_set=="gallery":
        # each gallery identity has 10 samples, embeddings will be saved based on the identity
        args.batch_size_perImg = (args.batch_size_perImg // 10) * 10
    else:
        # each image has different number of faces to be extracted, embeddings will be saved based on the image
        args.batch_size_perImg = 1

    inf_dataset = face_normalized(
        data,
        which_set=args.which_set,
        image_size=112, 
        align=True, 
        transform=trans
    )

    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=args.batch_size_perImg,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False)

    return inf_loader

def build_model(args):
    """
    It builds the MagFace model
    """
    from MagFace.inference.network_inf import builder_inf

    # magface requires cpu_mode argument
    args.cpu_mode = not args.gpu
    model = builder_inf(args)

    device = torch.device(f"cuda:{args.gpu[0]}" if args.gpu else "cpu")
    model = model.to(device)

    if len(args.gpu) > 1:
        model = torch.nn.DataParallel(model,device_ids=args.gpu)

    # switch to the evaluation mode
    model.eval()

    return model,device

def main(command_line_arguments=None):

    # get command line arguments
    args = command_line_options(command_line_arguments)

    # load extraction protocol
    logger.info("Loading UCCS %s extraction protocol", args.which_set)

    # read the detections for gallery,valid or test
    if args.which_set =="gallery":
        data = read_ground_truth(args.data_directory,"gallery")
    else:
        data = read_detections(os.path.join(args.data_directory,args.detection_file))
        
    landmarks = [l for _,_,l in data.values()]
    img_files = list(data.keys())

    # create the path of images, it differs based on the protocol because the gallery has sub-directories
    img_paths = [os.path.join(args.data_directory,args.image_directory,file.split("_")[0],file) for file in img_files] if args.which_set == "gallery"  else [
        os.path.join(args.data_directory,args.image_directory,file) for file in img_files]
    
    # download MagFace repo and its model weights if it wasnt downloaded before
    logger.info("Downloading/Activating MagFace and its model weights")
    download_MagFace(args)

    # get the data loader
    logger.info("Creating the %s dataloader", args.which_set)
    inf_loader = inference_loader((img_paths,None,landmarks),args)

    # get the model in eval mode
    logger.info("Loading the baseline model")
    model,device = build_model(args)

    saving_path = os.path.join(args.data_directory,args.feat_dir)
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
            _ = save_features_information(data,img_names,args.which_set,_feat,saving_path,args.batch_size_perImg)

if __name__ == "__main__":
    main()
