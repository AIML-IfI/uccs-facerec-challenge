# This file contains source code related to face normalization and feature extraction
import sys
import numpy as np
import torch.utils.data as data
import torch
from torchvision import transforms
import cv2
from skimage import transform
import os
import subprocess
import gdown

def download_MagFace(args,logger):

    """
    It downloads the MagFace model and its model weights unless it was downloaded before
    """

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
    args.baseline_recognition.resume = weights_path

class ImgData(data.Dataset):
    """
    It takes data containing the paths of images, bounding boxes/landmarks to align/crop the faces
    """
    def __init__(self, data,which_set=None,image_size=112, align=True, transform=None):
        
        self.img_paths,bboxes,landmarks = data
        self.points = landmarks if align else bboxes

        self.which_set = which_set

        if align:
            assert self.points != None
            self.align = align
            self.mode = "arcface"
            self.arcface_src = np.array(
                                [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                                [41.5493, 92.3655], [70.7299, 92.2041]],
                                dtype=np.float32)
            self.arcface_src = np.expand_dims(self.arcface_src, axis=0)

                                    
        self.transform = transform
        self.image_size = image_size

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
    
    def __len__(self):
        return len(self.img_paths)


def inference_dataloader(data,which_set,batch_size_perImg,workers):
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

    if which_set=="gallery":
        # each gallery identity has 10 samples, embeddings will be saved based on the identity
        batch_size_perImg = (batch_size_perImg // 10) * 10
    else:
        # each image has different number of faces to be extracted, embeddings will be saved based on the image
        batch_size_perImg = 1

    inf_dataset = ImgData(
        data,
        which_set=which_set,
        image_size=112, 
        align=True, 
        transform=trans
    )

    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size_perImg,
        num_workers=workers,
        pin_memory=True,
        shuffle=False)

    return inf_loader

def build_model(args):
    """
    It builds the MagFace model
    """
    from .MagFace.inference.network_inf import builder_inf

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

def save_features_information(data,img_names,which_set,features,save_directory,batch_size_perImg):
    """
    It saves all information including detection_score,bboxes,landmarks and embedding as .pth file
    """

    if which_set == "gallery":
        # save the all information based on the each identity
        for idx in range(0,batch_size_perImg,10):

            identity = img_names[idx].split("_")[0]
            landmarks = np.stack([data[name][1][0] for name in img_names[idx:idx+10]])

            image_values = {
                "landmarks": landmarks,
                "embeddings": features[idx:idx + 10, :]
            }
            
            torch.save(image_values,os.path.join(save_directory,f"{identity}.pth"))

    else:
        # otherwise, save the all information of the each image
        image_values = {
            "detection_scores": data[img_names[0]][0],
            "bboxes": data[img_names[0]][1],
            "landmarks": data[img_names[0]][2],
            "embeddings": features
        }

        torch.save(image_values,os.path.join(save_directory,f"{img_names[0][:-4]}.pth"))

