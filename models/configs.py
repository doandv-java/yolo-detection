# Config YOLO
YOLO_MODEL = 'v3'
YOLO_V3_WEIGHTS = "./data/weights/yolov3.weights"
YOLO_V4_WEIGHTS = "./data/weights/yolov4.weights"
YOLO_SAVE_WEIGHTS = "./data/saved_weights/yolo_{}.h5".format(YOLO_MODEL)
YOLO_INPUT_SIZE = 416
YOLO_CLASSES = "./data/classes/coco.names"
YOLO_STRIDES = [8, 16, 32]
YOLO_ANCHOR_PER_SCALE = 3
YOLO_MAX_BBOX_PER_SCALE = 100
YOLO_IOU_LOSS_THRESH = 0.5
if YOLO_MODEL == "v4":
    YOLO_ANCHORS = [[[12, 16], [19, 36], [40, 28]],
                    [[36, 75], [76, 55], [72, 146]],
                    [[142, 110], [192, 243], [459, 401]]]
    YOLO_WEIGHTS = YOLO_V4_WEIGHTS
if YOLO_MODEL == "v3":
    YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[116, 90], [156, 198], [373, 326]]]
    YOLO_WEIGHTS = YOLO_V3_WEIGHTS

# Config train
TRAIN_SAVE_BEST_ONLY = True  # saves only best model according validation loss (True recommended)
TRAIN_SAVE_CHECKPOINT = False  # saves all best validated checkpoints in training process (may require a lot disk space) (False recommended)
TRAIN_CLASSES = "data/classes/custom.names"
TRAIN_ANNOT_PATH = "data/train/train.txt"
TRAIN_LOGDIR = "log"
TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
TRAIN_MODEL_NAME = "yolov3_custom"
TRAIN_LOAD_IMAGES_TO_RAM = False  # faster training, but need more RAM
TRAIN_BATCH_SIZE = 8
TRAIN_INPUT_SIZE = 416
TRAIN_DATA_AUG = True
TRAIN_TRANSFER = False
TRAIN_FROM_CHECKPOINT = False  # "checkpoints/yolov3_custom"
TRAIN_LR_INIT = 1e-4
TRAIN_LR_END = 1e-6
TRAIN_WARMUP_EPOCHS = 2
TRAIN_EPOCHS = 100

# TEST options
TEST_ANNOT_PATH = "data/train/val.txt"
TEST_BATCH_SIZE = 4
TEST_INPUT_SIZE = 416
TEST_DATA_AUG = False
TEST_DECTECTED_IMAGE_PATH = ""
TEST_SCORE_THRESHOLD = 0.3
TEST_IOU_THRESHOLD = 0.45
