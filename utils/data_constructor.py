"""
    Function: Loading video samples. (1) Step by Step (YES), (2) Loading all data once (NO),
              since there is limited GPUs.

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 25, 2021
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import time
import os.path as osp


class ReadVideo(data.Dataset):
    """
    extracting video frames from a video.
    TODO: The code will be revised using the multi-processor technique to read image quickly.
    """
    def __init__(self, args):
        super(ReadVideo, self).__init__()
        self.num_frames = args['num_frames']  # the number of video frames extracted.
        self.img_size = args['img_size']
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=args['mean'], std=args['std'])])

    def extract_video_frames(self, video_name, videos_dir, txt_dir):
        video_path = osp.join(videos_dir, video_name)
        txt_path = osp.join(txt_dir, video_name.split(".")[0] + ".txt")

        id1_bboxes = []
        id2_bboxes = []
        with open(txt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                info = line.strip().split(" ")
                frame_index = int(info[0])
                id = int(info[1])
                x1 = int(info[2])
                y1 = int(info[3])
                x2 = int(info[4])
                y2 = int(info[5])
                if id == 1:
                    id1_bboxes.append([frame_index, x1, y1, x2, y2])
                else:
                    id2_bboxes.append([frame_index, x1, y1, x2, y2])

        total_frames = len(id1_bboxes)
        frame_indexes = [(i, (total_frames // self.num_frames) * i) for i in range(0, self.num_frames)]

        # self.cap.open(video_path)
        cap = cv2.VideoCapture(video_path)
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # read frames sequentially.
        # img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        img_height = self.img_size[0]
        img_width = self.img_size[1]

        assert img_width % 2 == 0, "the width of a region of interest must be divided by 2."
        sub_img_width = img_width // 2

        video_tensor = torch.zeros(self.num_frames, 3, img_height, img_width)
        for index, frame_index in frame_indexes:
            frame = id1_bboxes[frame_index][0]

            id1_x1 = id1_bboxes[frame_index][1]
            id1_y1 = id1_bboxes[frame_index][2]
            id1_x2 = id1_bboxes[frame_index][3]
            id1_y2 = id1_bboxes[frame_index][4]

            id2_x1 = id2_bboxes[frame_index][1]
            id2_y1 = id2_bboxes[frame_index][2]
            id2_x2 = id2_bboxes[frame_index][3]
            id2_y2 = id2_bboxes[frame_index][4]

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ret, img = cap.read()
            if ret is False:
                break

            id1_img = img[id1_y1:id1_y2, id1_x1:id1_x2]
            id2_img = img[id2_y1:id2_y2, id2_x1:id2_x2]

            id1_img = cv2.resize(id1_img, (sub_img_width, self.img_size[0]))
            id2_img = cv2.resize(id2_img, (sub_img_width, self.img_size[0]))

            img = np.concatenate((id1_img, id2_img), axis=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # H x W x C, [0, 255]
            
            # other data augmentation methods???
            img_tensor = self.transform(img)  # C x H x W, scaling.
            video_tensor[index] = img_tensor
        return video_tensor  # [num_frames, channels, height, width]


class DatasetConstructor(data.Dataset):
    """
    read videos to form a PyTorch supported data format. [batchsize, num_frames, channel, height, width]
    """
    def __init__(self, args, data_dir_path, videos_dir, txt_dir, num, is_train=True):
        """
        :param args:
        :param data_dir_path: recording video names and corresponding classes.
        :param videos_dir: storing all training/validation videos.
        :param txt_dir: storing all bounding boxes and corresponding person IDs.
        :param num: the number of training or validation videos.
        :param is_train:
        """
        super(DatasetConstructor, self).__init__()
        self.train = is_train
        self.txt_dir = txt_dir  # mutual_cross_setup_detection_tracking_together

        if self.train:
            self.train_data_gt_root = data_dir_path
            self.train_num = num
            self.train_videos = []
            self.train_permulation = np.random.permutation(self.train_num)
            self.train_videos_dir = videos_dir
        else:
            self.val_data_gt_root = data_dir_path
            self.val_num = num
            self.val_videos = []
            self.val_permulation = np.asarray(list(range(0, self.val_num)))
            self.val_videos_dir = videos_dir

        # just recording training and validation data paths pair.
        if self.train:
            with open(self.train_data_gt_root, "r") as train_data_gt_file:
                for line in train_data_gt_file:
                    train_data_gt = line.strip().split(" ")
                    train_video = train_data_gt[0]
                    train_gt = train_data_gt[1]
                    self.train_videos.append([train_video, train_gt])
        else:
            with open(self.val_data_gt_root, "r") as val_data_gt_file:
                for line in val_data_gt_file:
                    val_data_gt = line.strip().split(" ")
                    val_video = val_data_gt[0]
                    val_gt = val_data_gt[1]
                    self.val_videos.append([val_video, val_gt])

        # data preprocessing.
        self.read_video = ReadVideo(args=args)

    def __getitem__(self, index):
        start = time.time()
        if self.train:
            train_video = self.read_video.extract_video_frames(self.train_videos[self.train_permulation[index]][0],
                                                               self.train_videos_dir,
                                                               self.txt_dir)
            train_gt = int(self.train_videos[self.train_permulation[index]][1])
            end = time.time()
            return self.train_permulation[index]+1, train_video, train_gt, (end - start)
        else:
            val_video = self.read_video.extract_video_frames(self.val_videos[self.val_permulation[index]][0],
                                                             self.val_videos_dir,
                                                             self.txt_dir)
            val_gt = int(self.val_videos[self.val_permulation[index]][1])
            end = time.time()
            return self.val_permulation[index]+1, val_video, val_gt, (end - start)

    def __len__(self):
        if self.train:
            return self.train_num
        else:
            return self.val_num

    def shuffle(self):
        if self.train:
            self.train_permulation = np.random.permutation(self.train_num)
        else:
            self.val_permulation = np.asarray(list(range(0, self.val_num)))
        return self

    def val_mode(self):
        self.train = False
        return self

    def train_mode(self):
        self.train = True
        return self


if __name__ == "__main__":
    # test the ReadVideo class.
    from config import parser
    args = vars(parser.parse_args())
    read_video = ReadVideo(args)
    # video_path = "E:/CodeDatasets/Datasets/DAiSEE/DataSet/Train/110001/1100011002/1100011002.avi"
    video_name = "xxx.avi"
    video_dir = "E:/CodeDatasets/"
    txt_dir = "E:/CodeDatasets/"
    video_tensor = read_video.extract_video_frames(video_name=video_name,
                                                   videos_dir=video_dir,
                                                   txt_dir=txt_dir)
    print(video_tensor.shape)
    # print(video_tensor)

    # test the DataConstructor class. TODO: when training.

    # data_dir_path = "x/data/train.txt"  # "data/test.txt"
    # num = 30  # 3626 for training, and 2782 for test.
    # is_train = True
    # train_dataloader = DatasetConstructor(data_dir_path=data_dir_path, num=num, is_train=is_train)
    #
    # index, train_img, train_gt, ti = train_dataloader.__getitem__(1)
    # train_len = train_dataloader.__len__()
    # print(index, train_img.shape, train_gt.shape, ti, train_len)
    # print(torch.unique(train_gt))
    #
    # print("Successfully!")

