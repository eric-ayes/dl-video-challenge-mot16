import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

_sets = {}

# Fill all available datasets, change here to modify / add new datasets.
for split in ['train', 'test', 'all', '01', '02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14','95','96','97','98','99']:
    name = f'MOT16-{split}'
    _sets[name] = (lambda root_dir, split=split, **kwargs: MOT16(root_dir, split, **kwargs))


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


class MOT16Sequences():
    """
    A central class to manage the individual dataset loaders.

    This class contains references to train and test datasets.
    Once initialized the individual parts (e.g. sequences) can be accessed be using the indexing operator.
    """
    def __init__(self, dataset: str, root_dir: str, **kwargs):
        """
        Initialize a `MOT16Sequences` making it able to load data from train and test set.

        :param str dataset: The name of the dataset or sequence to load.
        :param str root_dir: The root directory pointing to the dataset.
        :param kwargs: Arguments used to call the dataset.
        """
        assert dataset in _sets, "[!] Dataset not found: {}".format(dataset)

        self._data = _sets[dataset](root_dir, **kwargs)

    def __len__(self) -> int:
        """
        Get length of dataset.

        :return int: Length of dataset.
        """
        return len(self._data)

    def __getitem__(self, idx):
        """
        Retrieve dataset item at index `idx`.

        :param int idx: Idx of item to be retrieved.

        :return: The dataset item at index `idx`.
        """
        return self._data[idx]


class MOT16(Dataset):
    """
    A Wrapper for the MOT_Sequence class to return multiple sequences.
    """
    def __init__(self, root_dir, split, **kwargs):
        """
        Initliazes all subset of the dataset.

        :param root_dir: Directory of the dataset.
        :param split: The split of the dataset to use.
        :param kwargs: Keyword arguments used to call the dataset.
        """
        train_sequences = list(listdir_nohidden(os.path.join(root_dir, 'train')))
        test_sequences = list(listdir_nohidden(os.path.join(root_dir, 'test')))

        if "train" == split:
            sequences = train_sequences
        elif "test" == split:
            sequences = test_sequences
        elif "all" == split:
            sequences = train_sequences + test_sequences
        elif f"MOT16-{split}" in train_sequences + test_sequences:
            sequences = [f"MOT16-{split}"]
        else:
            raise NotImplementedError("MOT split not available.")

        self._data = []
        for s in sequences:
            self._data.append(MOT16Sequence(root_dir, seq_name=s, **kwargs))

    def __len__(self):
        """
        Get length of dataset.

        :return int: Length of dataset.
        """
        return len(self._data)

    def __getitem__(self, idx):
        """
        Retrieve dataset item at index `idx`.

        :param int idx: Idx of item to be retrieved.

        :return: The dataset item at index `idx`.
        """
        return self._data[idx]


class MOT16Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataset is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """
    def __init__(self, root_dir, seq_name, vis_threshold=0.0, load_seg=False):
        """
        Initialize the MOT16 Sequence dataset.
        
        :param root_dir: Directory of the dataset.
        :param string seq_name: Sequence to take.
        :param float vis_threshold: Threshold of visibility of persons above which they are selected.
        """
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold
        self._load_seg = load_seg
        self._mot_dir = root_dir

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        assert seq_name in self._train_folders or seq_name in self._test_folders, \
            'Image set does not exist: {}'.format(seq_name)

        self.data, self.no_gt = self._sequence()

    def __len__(self) -> int:
        """
        Return the length of the sequence.

        :return int: Length of the sequence.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the image at `idx` with additional information.
        
        :param int idx: Index of the image to be returned.
        
        :return dict: The image and additional information at index `idx`.
        """
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        # ----------------------------
        # YOLO preprocessing
        # ----------------------------
        # gt_boxes = list(data['gt'].values())
        # gt_boxes = np.stack(gt_boxes) if gt_boxes else np.zeros((0, 4), dtype=np.float32)

        # if self.yolo_model:
        #     img, gt_boxes = self.letterbox_image_and_boxes(img, gt_boxes, self.yolo_size)

        sample = {
            'img': img,
            'img_path': data['im_path'],
            'gt': data['gt'], #gt_boxes,
            'vis': data['vis']
        }

        # sample = {}
        # sample['img'] = img
        # sample['img_path'] = data['im_path']
        # sample['gt'] = data['gt']
        # sample['vis'] = data['vis']

        # segmentation
        if data['seg_img'] is not None:
            seg_img = np.array(data['seg_img'])
            # filter only pedestrians
            class_img = seg_img // 1000
            seg_img[class_img != 2] = 0
            # get instance masks
            seg_img %= 1000
            sample['seg_img'] = seg_img

        return sample
    
    # def letterbox_image_and_boxes(self, img: torch.Tensor, boxes: np.ndarray, target_size=640):
    #     """
    #     Resize and pad image to target_size for YOLO, adjusting GT boxes accordingly.
    #     img: [C,H,W] tensor
    #     boxes: [N,4] array of [x1,y1,x2,y2]
    #     """
    #     _, h, w = img.shape
    #     scale = min(target_size / w, target_size / h)
    #     new_w = int(w * scale)
    #     new_h = int(h * scale)

    #     pad_w = target_size - new_w
    #     pad_h = target_size - new_h
    #     pad_left = pad_w // 2
    #     pad_top = pad_h // 2

    #     # Resize image
    #     img = torch.nn.functional.interpolate(
    #         img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
    #     ).squeeze(0)

    #     # Pad to target size
    #     img = torch.nn.functional.pad(img, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0)

    #     # Adjust GT boxes
    #     if boxes.shape[0] > 0:
    #         boxes = boxes * scale
    #         boxes[:, [0, 2]] += pad_left
    #         boxes[:, [1, 3]] += pad_top

    #     return img, boxes

    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._mot_dir, 'train', seq_name)
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        img_dir = config['Sequence']['imDir']

        img_dir = osp.join(seq_path, img_dir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')
        seg_dir = osp.join(seq_path, 'seg_ins')

        data = []
        boxes = {}
        visibility = {}
        seg_imgs = {}

        for i in range(1, seqLength+1):
            boxes[i] = {}
            visibility[i] = {}

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(row[2]) - 1
                        y1 = int(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(row[4]) - 1
                        y2 = y1 + int(row[5]) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        if self._load_seg:
            if osp.exists(seg_dir):
                for seg_file in listdir_nohidden(seg_dir):
                    frame_id = int(seg_file.split('.')[0])
                    seg_img = Image.open(
                        osp.join(seg_dir, seg_file))
                    seg_imgs[frame_id] = seg_img

        for i in range(1, seqLength + 1):
            img_path = osp.join(img_dir, f"{i:06d}.jpg")

            datum = {'gt': boxes[i],
                     'im_path': img_path,
                     'vis': visibility[i]}

            datum['seg_img'] = None
            if seg_imgs:
                datum['seg_img'] = seg_imgs[i]

            data.append(datum)


        return data, no_gt

    def __str__(self):
        """
        String representation of the sequence.
        """
        return self._seq_name

    def write_results(self, all_tracks, output_dir):
        """
        Write the tracks in the format for MOT16/MOT17 sumbission.

        Each file contains these lines:
        `<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>`

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt

        :param all_tracks: Dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num.
        :param output_dir: Directory where to save the results.
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, 'MOT16-'+self._seq_name[6:8]+'.txt')

        print("Writing predictions to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
