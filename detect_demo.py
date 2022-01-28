import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from models.yolo import create_yolo
from models.configs import *
from models import utils

yolo = create_yolo(YOLO_INPUT_SIZE, classes=YOLO_CLASSES, model=YOLO_MODEL)
yolo.load_weights(YOLO_SAVE_WEIGHTS)
utils.detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE)
