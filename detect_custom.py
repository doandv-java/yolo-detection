import os
from models.yolo import create_yolo
from models.configs import *
from models import utils
from datetime import datetime

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# input
video_test_name = "test"
video_path = "data/test/{}.mp4".format(video_test_name)
train_model_name = "yolov3_custom"
yolo_weights = "./checkpoints/{}".format(train_model_name)
result_test_root = "data/result_test"
iou = 0.5
score = 0.5
# out put
result_test_checkpoint_folder = os.path.join(result_test_root, train_model_name)
if not os.path.exists(result_test_checkpoint_folder):
    os.makedirs(result_test_checkpoint_folder, exist_ok=True)
result_test_video_folder = os.path.join(result_test_checkpoint_folder, video_test_name)
if os.path.exists(result_test_video_folder):
    os.makedirs(result_test_video_folder, exist_ok=True)
detect_time = datetime.now()
str_detect_time = detect_time.strftime("%m%d%Y_%H%M%S")
output_name = f"{video_test_name}_{score * 10}_{iou * 10}_{str_detect_time}.avi"
output_path = os.path.join(result_test_video_folder, output_name)

# model
yolo = create_yolo(input_size=YOLO_INPUT_SIZE, classes=TRAIN_CLASSES)
yolo.load_weights(yolo_weights)  # use keras weights
# detect
utils.detect_video(yolo, video_path, output_path, input_size=YOLO_INPUT_SIZE, show=True, classes=TRAIN_CLASSES,
                   score_threshold=score, iou_threshold=iou)
