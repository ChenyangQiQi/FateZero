import os
import logging, logging.handlers
from accelerate.logging import get_logger

def get_logger_config_path(logdir):
    # accelerate handles the logger in multiprocessing
    logger = get_logger(__name__)
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s:%(levelname)s : %(message)s', 
        datefmt='%a, %d %b %Y %H:%M:%S', 
        filename=os.path.join(logdir, 'log.log'),
        filemode='w')
    chlr = logging.StreamHandler()
    chlr.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s : %(message)s'))
    logger.logger.addHandler(chlr)
    return logger