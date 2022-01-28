from models.yolo import create_yolo
from models.configs import *
from models import utils

if __name__ == '__main__':
    yolo = create_yolo(input_size=YOLO_INPUT_SIZE, classes=YOLO_CLASSES, model=YOLO_MODEL)
    utils.load_yolo_weights(yolo, YOLO_WEIGHTS)
    yolo.summary()
    yolo.save_weights(YOLO_SAVE_WEIGHTS)
    print('Done')
