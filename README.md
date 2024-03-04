# UCCS Watchlist Challenge

This package utilizes the PyTorch framework to implement baseline algorithms and conduct evaluations for the third version of the Open-set Face Detection and Identification challenge.

## Dataset
----------

This package does not include the original image and protocol files for the competition.
Please register on the [Competition Website](https://www.ifi.uzh.ch/en/aiml/challenge.html). Upon confirming your registration, we will provide the data, including gallery and validation images along with their protocol files, in a zip file. Please extract zip files **into a new directory named `data`** (the .zip files contain the appropriate directory structure) within this package. After this, the directory structure should appear as follows:

    uccs-facerec-challenge/
    ├── data/
    │   ├── gallery_images/
    │   │     ├── 0001/
    │   │     │   ├── 0001_0.png
    │   │     │   ├── ...
    │   │     │   └── 0001_9.png
    │   │     ├── ...
    │   │     └── 1000
    │   │
    │   ├── validation_images/
    │   │   ├── jgsa451150sag15fou.jpg
    │   │   ├── ...
    │   │
    │   ├── exclude_gallery_validation.txt
    │   ├── gallery.csv
    │   └── validation.csv
    │
    ├── facerec/
    │   ├── configs/
    │   ├── script/
    │   └── ...
    │
    ├── environment.yaml
    ├── README
    ├── setup.py
    └── ...

If you prefer to store this data in a different directory from `data`, you should modify ``--data_directory`` in the `facerec/configs/baseline_config.yaml` file or overwrite it on the command line:

```bash
    [script] --data_directory YOUR-DATA-DIRECTORY
```
The test set images without any annotations will be distributed two weeks before the competition concludes, as stated on the [Competition Website](https://www.ifi.uzh.ch/en/aiml/challenge.html).

## Installation
---------------

The installation of this package follows via conda:

```bash
    git clone https://github.com/AIML-IfI/uccs-facerec-challenge.git
    cd uccs-facerec-challenge
    conda env create -f environment.yaml
    conda activate uccs-facerec
```

## Scripts
-----------

There are four scripts in total, which can be found in `facerec/script`. 
All scripts will be installed together with the installation of this package in order to run the baseline algorithms or to evaluate the baselines (and your) algorithm.
The networks employed in our baselines will be automatically downloaded when you run the corresponding script. Hence, there is no need for you to separately download these baseline algorithms which are [MTCNN](https://pypi.org/project/facenet-pytorch/) and [MagFace](https://github.com/IrvingMeng/MagFace).

Each script is associated with its distinct set of parameters specified in the configuration *.yaml* file, located at `facerec/configs/baseline_config.yaml`.
In the `baseline_config.yaml` file, you can find all arguments, along with their default values, under their respective sections.
Nonetheless, there are four essential arguments placed at the top of the configuration file that are required for almost all scripts. 

Here are these options that you might want/need to use/change:

  ``--data_directory``: Specify the directory, where you have downloaded the UCCS dataset into; default: ``data``.

  ``--result_directory``: Specify the directory, where the score files, evaluation curves, and embeddings will be stored; default: ``results``. Note that this directory doesn't need to exist.

  ``--which_set``: The set, for which the scripts should be run; possible values: ``validation, test``; default: ``validation``.

  ``--image_directory``: Specify the directory, where the original images are stored; default: ``{data_directory}/{which_set}_images``
  
  ``--gpu``: The GPU index to run the detector/extractor on it, to run in CPU mode please specify ``-1``; default: ``0``.


You have the option to adjust these parameters within the configuration file or generate a new configuration file using the same format. Make sure that the newly created configuration file is explicitly specified on the command line:

```bash
    [script] YOUR-CONFIG-FILE
```

It's also possible to overwrite any parameter on the command line. In this case, arguments you wish to overwrite should be stated on the command line after calling the script:

```bash
    [script] --param1 YOUR-VALUE --param2 YOUR-VALUE
```

The detailed explanations for each script are provided within their corresponding sections below.

### Face Detection
The first script is a face detection script, which will detect the faces in the validation (and test) set and write the results (detection scores, bounding boxes and landmarks) into a file.
The bounding boxes consist of four components—left, top, width, and height—while the detection scores fall within the range of 0 to 1.
Facial landmarks include (x,y) coordinates for the right eye, left eye, nose, right mouth, left mouth, respectively.

The baseline detector uses the PyTorch implementation of the [MTCNN](https://pypi.org/project/facenet-pytorch/) face detection module, where we had to lower the detection thresholds in order to detect most of the faces in the images, but this leads to many background detections.

You can easily call the face detector baseline script after successful installation using:

  ``baseline_detection``

If the baseline configuration file suits your environment, there's no need to specify the configuration file's path on the command line while calling the script. Simply running the script will automatically read the default `baseline_config.yaml` file.

Here are the options that you might want/need to use/change for this script:

  ``--detection.results``: Specify the file to write the detection results into; default: ``{result_directory}/UCCS-detection-baseline-{which_set}.txt``.

  ``--detection.parallel``: Run in the given number (positive integer) of parallel processes; default: ``0``

> **Note that** To achieve the same baseline detection results, the ``--detection.thresholds : [0.2,0.2,0.2]`` and ``--detection.max_detections: 50`` options should remain unchanged.

Here is an example of how to overwrite any parameter in the configuration file using the command line:

  ```bash
  baseline_detection  --data_directory YOUR-DATA-DIRECTORY --detection.results YOUR-FILE-PATH
  ```

### Face Recognition

For face recognition, we utilize the [MagFace](https://github.com/IrvingMeng/MagFace) (the backbone of iResNet-100), a cutting-edge method in the field of face recognition to extract features from detected faces above.
The MagFace model and its weights don't require any manual downloading from the Internet; they are automatically downloaded when the script is called for the first time.

You can easily call the face extractor baseline script after successful installation using:

  ``baseline_recognition``

The MagFace extracts features with the shape of (,512) for both probe and gallery. 
For the validation, the script stores all information (detection scores, bounding boxes, and embeddings) of the faces based on the probe image.
The bounding boxes consist of four components—left, top, width, and height—while the detection scores fall within the range of 0 to 1.
For the gallery, it stores five facial landmarks (including (x,y) coordinates for right eye, left eye, nose, right mouth, left mouth) and embeddings based on an identity. In conclusion, it compiles these dictionaries into .pth files for both sets, simplifying the process of generating the score file in the next step.

Once the script execution is complete, the directory structure appears as follows:

    results/
    ├── validation_embeddings/
    │   ├── probe-image1.pth
    │   │     - Content (a dict): {
    │   │         'detection_scores': Array with shape (N, 1), # where N is the number of detected faces in the image
    │   │         'bboxes': Array with shape (N, 4)
    │   │         'embeddings': Array with shape (N, 512)
    │   │         }
    │   │
    │   ├── probe-image2.pth
    │   ├── ...
    │
    ├── gallery_embeddings/
        ├── subject1.pth         # each subject has 10 faces in the gallery
        │     - Content (a dict):{   
        │         'landmarks': Array with shape (10, 5,2) 
        │         'embeddings': Array with shape (10,512)
        │           }
        ├── subject2.pth
        ├── ...

If the baseline configuration file suits your environment, there's no need to specify the configuration file's path on the command line while running the script. Simply calling the script will automatically read the `baseline_config.yaml` file.

Here are the options that you might want/need to use/change for this script:

  ``--disable-gallery``: When specified,  it will disable the gallery extraction with the given set at once; default: ``False``

  ``--recognition.detection_file``: The file containing the detected faces in the images; default: ``{result_directory}/UCCS-detection-baseline-{which_set}.txt``.

  ``--recognition.result_dir``: Specify the directory to store .pth files of the probe images/subjects; default: ``{result_directory}/{which_set}_embeddings``

  ``--recognition.workers``: How many sub-processes to use for data loading; default: ``4``

  ``--recognition.batch_size_perImg``: For validation (or test); it should be 1 because of the different number of faces in each image; for gallery, it can be multiples of 10; default: ``1``. Even if you don't provide a proper value for the batch size, the script is able to handle it.

> **Note that** To achieve the same baseline recognition results, the ``--recognition.embedding_size: 512`` and ``--recognition.arch: iresnet100`` options should remain unchanged.

Here is an example of how to overwrite any parameter in the configuration file using the command line:

  ```bash
  baseline_recognition --recognition.workers NUMBER-OF-WORKERS --recognition.result_dir YOUR-RESULT-DIRECTORY
  ```

### Scoring

This script produces the desired score file for the last evaluation phase, applicable for both face detection and identification evaluations.
For smooth execution of this scoring process, it is essential to store the embeddings of the gallery and validation, as explained in the preceding section (face recognition).
This scoring initially reads gallery embeddings to establish enrollment, achieved by averaging 10 embeddings of the corresponding subject. 
Therefore, each subject is represented by an array with a shape of (1,512) in the enrollment. 
Following the enrollment, the script compares the embedding of each face in the probe images with those of 1000 subjects using cosine similarity. 
Finally, it writes each similarity score of every subject along with their detection results (confidence scores and bounding boxes (left, top, width, height)) into a file.

> **Note that** If your intention is only to participate in the face detection task, calling this script is unnecessary. Creating a file similar to ``baseline_detection`` is sufficient. Further details can be found on the [Competition Website](https://www.ifi.uzh.ch/en/aiml/challenge.html) regarding the expected format of score files.

You can easily call the scoring script after successful installation using:

  ``scoring``

If the baseline configuration file suits your environment, there's no need to specify the configuration file's path on the command line while running the script. Simply calling the script will automatically read the default `baseline_config.yaml` file.

Here are the options that you might want/need to use/change for this script:

  ``--scoring.gallery``: Specify the directory where the gallery embeddings (.pth files) are stored; default: ``{result_directory}/gallery_embeddings``.

  ``--scoring.probe``: Specify the directory where the probe embeddings (.pth files) are stored; default: ``{result_directory}/{which_set}_embeddings``.

  ``--scoring.results``: Specify the file to write the scores and detection results into; default: ``{result_directory}/UCCS-scoring-baseline-{which_set}.txt``.

Here is an example of how to overwrite any parameter in the configuration file using the command line:

  ```bash
  scoring --scoring.gallery GALLERY-EMBEDDINGS-DIRECTORY --scoring.probe PROBE-EMBEDDINGS-DIRECTORY
  ```

### Evaluation

The provided evaluation script is usable to evaluate the validation set only, not the test set (since the test set labels will not be given to the participants). 
By default, this script is capable of running the evaluation for both face detection and identification. 
If you are unable to run the baseline experiments on your machine, we will provide the baseline score files for the validation set alongside the dataset when shared.

You can use the evaluation script for two purposes:

1. To plot the baseline results in comparison to your results.
2. To make sure that your score file is in the desired format.

You can easily call the evaluation script after successful installation using:

  ``evaluation``

If the baseline configuration file suits your environment, there's no need to specify the configuration file's path on the command line while running the script. Simply calling the script will automatically read the default `baseline_config.yaml` file.

Here are the main options that you might want/need to use/change for this script:

  ``--tasks``: Specify the tasks that will perform in this evaluation; possible values: ``detection, recognition``; default : ``['detection', 'recognition']``.

  ``--eval.exclude_gallery``:  Specify the file where gallery face IDs are stored to exclude them from the results; default: ``{data_directory}/exclude_gallery_{which_set}.txt``. Note that since the gallery faces are cropped from the dataset, they will be excluded from the results. This list of gallery face IDs will be shared along with the dataset.

  ``--eval.linear``: If specified True, the False Positive / False Alarm axes will be plotted in linear form, otherwise in logaritmic form; default : ``False``.

> **Note that** If you plan to participate in both challenges, the face recognition score file can be used for evaluating both the detection and the recognition experiment. Therefore, it is enough to only submit the desired identification score file. More details can be found in the [Competition Website](https://www.ifi.uzh.ch/en/aiml/challenge.html).

#### Detection

In the face detection evaluation, the script compares detected bounding boxes, utilizing the standard IOU metric with a threshold of `0.5`, against the ground truth. 
Only the detection box with the highest overlap can be considered a true positive, while others are penalized. 
The evaluation employs the Free Receiver Operator Characteristic (FROC) curve, ploting the Detection Rate (percentage of correctly detected faces) against False Detection per Image (detected background regions).
False Detection per Image is calculated by dividing the misdetections by the number of probe images.
Different points on the FROC curve can be obtained for different detector confidence values.

You can easily call the evaluation script for only detection task after successful installation using:

  ```bash
  evaluation --tasks detection
  ```
Here are the options that you might want/need to use/change for this detection task:

  ``--eval.detection.files``: A list of score file(s) containing detection results; default : ``['{result_directory}/UCCS-detection-baseline-{which_set}.txt']``. For comparison, different detection score files can be added. 

  ``--eval.detection.labels``: A list of label(s) for the algorithms; must be the same number and in the same order as ``--eval.detection.files``; default: ``['mtcnn']``.

  ``--eval.detection.froc``: The .pdf file where FROC curve will be written into; default : ``{result_directory}/UCCS-FROC-{which_set}.pdf``.

> **Note that** To achieve the same baseline FROC curve on the validation, the ``--eval.iou: 0.5`` option should remain unchanged.

Here is an example of how to overwrite any parameter in the configuration file using the command line:

   ```bash
  evaluation --tasks detection --eval.detection.files FILE1 FILE2 --eval.detection.labels LABEL1 LABEL2
  ```

#### Recognition

During the evaluation of face recognition models, faces will be either assigned to an identity or rejected based on these similarity scores. 
All unique similarity score values in the score file will be threshold values for matching.
A face is counted correctly identified if the recognition score surpasses the threshold, and the correct identity possesses the highest recognition score for that particular face.
Providing high scores for unknown identities or misdetections, which indicate a false match with a gallery identity, will result in penalties. 

The evaluation utilizes a modified version of the Detection and Identification Rate (DIR) [1] curve on rank 1, also known as the Open-Set ROC curve that computes True Positive Identification Rates (TPIR) over False Positive Identification Rates (FPIR). 
Since the false alarm axis is dependent on the number of detected faces, we make slight modifications to this axis by dividing it by the number of probe images, leading to the False Positive Identification per Image [2].
This x-axis is in a logarithmic scale, representing non-rejected unknown faces and misdetections.
To prevent an increase in False Identifications, these unknown faces or misdetections should have a similarity score lower than the threshold specified for the points on the curve.

You can easily call the evaluation script for only the recognition task after successful installation using:

  ```bash
  evaluation --tasks recognition
  ```
Here are the options that you might want/need to use/change for this recognition task:

  ``--eval.recognition.files``: A list of score file(s) containing recognition results; default : ``['{result_directory}/UCCS-scoring-baseline-{which_set}.txt']``. For comparison, different recognition score files can be added.

  ``--eval.recognition.labels``: A list of the label(s) for the algorithms; must be the same number and in the same order as ``--eval.recognition.files``; default: ``['MagFace']``.

  ``--eval.recognition.rank``: Plot O-ROC curve(s) for the given rank; default : ``1``.

  ``--eval.recognition.oroc``: The .pdf file where O-ROC curve will be written into; default : ``{result_directory}/UCCS-OROC-{which_set}.pdf``.

> **Note that** To achieve the same baseline O-ROC curve on the validation, the ``--eval.iou: 0.5`` and ``--eval.recognition.rank: 1`` options should remain unchanged.

Here is an example of how to overwrite any parameter in the configuration file using the command line:

   ```bash
  evaluation --tasks recognition --eval.recognition.files FILE1 FILE2 --eval.recognition.labels LABEL1 LABEL2
  ```

## Trouble Shooting
-------------------

In case of trouble with running the baseline algorithm or the evaluation, please contact us via email: furkan.kasim@uzh.ch

[1] **P. Jonathon Phillips, Patrick Grother, and Ross Micheals** "Evaluation Methods in Face Recognition" in *Handbook of Face Recognition*, Second Edition, 2011.
[2] **Manuel Günther, Akshay Raj Dhamija and Terrance E. Boult.** Watchlist Adaptation: Protecting the Innocent. *International Conference of the Biometrics Special Interest Group (BIOSIG)*, 2020