# It generates the scoring .csv file before the evaluation.
import logging
import yamlparser
import os
from ..enrollment import average
from ..scoring import create_score_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceRec.UCCS")

def read_config_file():
    
    parent_direct = os.path.dirname(os.path.dirname(__file__))
    cfg = yamlparser.config_parser(default_config_files=[os.path.join(parent_direct, "configs/baseline_config.yaml")])

    if (cfg.scoring.gallery is None or cfg.scoring.probe is None):
        raise ValueError("For the scoring task, both --scoring.gallery and --scoring.probe are required.")
    
    return cfg

def main():

    # get config params
    cfg = read_config_file()

    logger.info("Loading UCCS %s scoring protocol",cfg.which_set)

    # get gallery enrollment
    logger.info("Getting UCCS gallery enrollment (average)")
    gallery_embedd_path = cfg.scoring.gallery
    subject_ids,gallery_enroll = average(cfg.data_directory,gallery_embedd_path)

    subject_ids = ["S_"+i for i in subject_ids]

    # compute scores between enrollment and probe and write them into a file
    probe_path = cfg.scoring.probe
    scoring_path = cfg.scoring.results
    logger.info("Computing scores and writing them into %s",scoring_path)

    _ = create_score_file((subject_ids,gallery_enroll),probe_path,scoring_path,cfg.gpu)

if __name__ == "__main__":
    main()