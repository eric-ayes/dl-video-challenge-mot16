import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchvision
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tracker.data_track import MOT16Sequences
from tracker.object_detector import FRCNN_FPN, YOLODetector
from tracker.reid_extractor import ReIDExtractor
from tracker.tracker import Tracker
from tracker.utils import evaluate_mot_accums, get_mot_accum

print(f"torch: {torch.__version__}, torchvision: {torchvision.__version__}, torchaudio: {torchaudio.__version__}")

working_dir = "/Users/marcelhofmann/Deep_Learning_UAM/Deep_Learning_for_Video_Signal_Processing/MOT_Challenge/dlvsp_challenge_material"

train_dir = os.path.join(working_dir,'data/MOT16/train')
test_dir = os.path.join(working_dir,'data/MOT16/test')
model_dir = os.path.join(working_dir,'models')

# config values
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Computation device: ', device)

#location of the model file
obj_detect_model_file = os.path.join(working_dir, "models/faster_rcnn_fpn.model")

#threshold for non maximum suppression
obj_detect_nms_thresh = 0.3

#detector has been trained for two classes
num_classes=2 # 1 class (person) + background (see https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

rpn_pre_nms_top_n_train=2000,
rpn_pre_nms_top_n_test=1000,
rpn_post_nms_top_n_train=2000,
rpn_post_nms_top_n_test=1000,

seed = 12345 #seed to allow repeatable results
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True

def get_detector_and_data_output_dir():
    # obj_detect_model_file = os.path.join(working_dir, "models/faster_rcnn_fpn.model") #location of the model file
    obj_detect_model_file = "/Users/marcelhofmann/Downloads/best_model_2.model" # frcnn_best_pedestrian.pth
    
    # obj_detect = Detector(img_size=(400, 600), out_size=(13, 19), out_channels=2048, n_classes=1, roi_size=(2, 2))
    obj_detect = FRCNN_FPN(
        num_classes=num_classes,  # person + background
        box_score_thresh=0.7,          # only keep high-confidence boxes
        box_nms_thresh=0.4,            # stricter NMS to remove overlapping boxes
        box_detections_per_img=50,     # max detections per frame
        rpn_score_thresh=0.05,         # optional, filter low-quality proposals early
        rpn_post_nms_top_n_test=500,   # limit proposals reaching box head
        min_size=600,                  # optional, reduce tiny detections
        rpn_pre_nms_top_n={
            "training": rpn_pre_nms_top_n_train,
            "testing": rpn_pre_nms_top_n_test,
        },
        rpn_post_nms_top_n={
            "training": rpn_post_nms_top_n_train,
            "testing": rpn_post_nms_top_n_test,
        },
    )

    obj_detect_state_dict = torch.load(obj_detect_model_file,map_location=lambda storage, loc: storage)
    obj_detect.load_state_dict(obj_detect_state_dict)
    obj_detect.eval()     # set to evaluation mode

    
    # obj_detect = YOLODetector(device=device)
    # obj_detect.model.eval()     # set to evaluation mode
    # obj_detect.model.to(device) # load detector to GPU or CPU
    print(f"The object detector has {sum(p.numel() for p in obj_detect.parameters())} trainable parameters.") #if p.requires_grad

    reid_extractor = ReIDExtractor(device)
    print(f"The ReID extractor has {sum(p.numel() for p in reid_extractor.model.parameters() if p.requires_grad)} trainable parameters.")

    # dataset
    seq_name = 'MOT16-train' # uncomment to run the tracker over the train set
    #seq_name = 'MOT16-test' # uncomment to run the tracker over the test set
    #seq_name = 'MOT16-02'   # uncomment to run the tracker over the sequence 'MOT16-02'
    data_dir = os.path.join(working_dir, 'data/MOT16')
    sequences = MOT16Sequences(seq_name, data_dir)
    print('Loaded {:d} sequences for {:s}'.format(len(sequences),seq_name))

    #output directory
    output_dir = os.path.join(working_dir, 'output')

    return obj_detect, reid_extractor, sequences, output_dir

def predict_and_track(tracker: Tracker, sequences: MOT16Sequences, output_dir: str):
    time_total = 0
    mot_accums = []
    results_seq = {}

    for seq in sequences:
        print(f"Tracking: {seq}")
        now = time.time()

        # restart tracker state for each sequence
        tracker.reset()

        #load data
        data_loader = DataLoader(seq, batch_size=1, shuffle=False) # batch size shouldnt be changed, no shuffling for tracking because logical order is needed

        #run tracker
        with torch.inference_mode(): # torch.no_grad()
            for frame in tqdm(data_loader):
                tracker.step(frame)

        #keep results
        results = tracker.get_results()
        results_seq[str(seq)] = results

        #perform evaluation
        if seq.no_gt:
            print(f"No GT evaluation data available.")
        else:
            mot_accums.append(get_mot_accum(results, seq)) #compute and store eval metrics

        time_total += time.time() - now

        print(f"Tracks found: {len(results)}")
        print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

        #save results to output directory
        seq.write_results(results, os.path.join(output_dir))

    print(f"Total tracking time for all sequences: {time_total:.1f} s.")
    return results_seq, mot_accums

def evaluate_results(mot_accums, sequences: MOT16Sequences, output_dir: str):
    if not hasattr(pd.DataFrame, "append"):
        pd.DataFrame.append = lambda self, other, **kwargs: pd.concat([self, other], **kwargs)

    if not hasattr(pd.Series, "iteritems"):
        pd.Series.iteritems = pd.Series.items
    if len(mot_accums) > 0:
        evaluate_mot_accums(
            mot_accums,
            [str(s) for s in sequences if not s.no_gt], 
            generate_overall=True
        )
    else:
        print("No evaluation performed, no GT data available.")

if __name__ == "__main__":
    obj_detect, reid_extractor, sequences, output_dir = get_detector_and_data_output_dir()
    tracker = Tracker(obj_detect,) # reid_extractor
    results_seq, mot_accums = predict_and_track(tracker, sequences, output_dir)
    evaluate_results(mot_accums, sequences, output_dir)