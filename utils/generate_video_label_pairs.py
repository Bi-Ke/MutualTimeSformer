"""
    Function: write paths of videos and corresponding labels to a file.

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 3, 2021
    Revised: April 25, 2021
"""

import os
import os.path as osp
import glob

label_name_dict = {"A050": "punch/slap", "A051": "kicking", "A052": "pushing", "A053": "pat_on_back", "A054": "point_finger",
                   "A055": "hugging", "A056": "giving_object", "A057": "touch_pocket", "A058": "shaking_hands", "A059": "walking_towards",
                   "A060": "walking_apart", "A106": "hit_with_object", "A107": "wield_knife", "A108": "knock_over", "A109": "grab_stuff",
                   "A110": "shoot_with_gun", "A111": "step_on_foot", "A112": "high-five", "A113": "cheers_and_drink", "A114": "carry_object",
                   "A115": "take_a_photo", "A116": "follow", "A117": "whisper", "A118": "exchange_things", "A119": "support_somebody",
                   "A120": "rock-paper-scissors"}

label_class_dict = {"A050": 0, "A051": 1, "A052": 2, "A053": 3, "A054": 4,
                    "A055": 5, "A056": 6, "A057": 7, "A058": 8, "A059": 9,
                    "A060": 10, "A106": 11, "A107": 12, "A108": 13, "A109": 14,
                    "A110": 15, "A111": 16, "A112": 17, "A113": 18, "A114": 19,
                    "A115": 20, "A116": 21, "A117": 22, "A118": 23, "A119": 24,
                    "A120": 25}


def generate_video_label_pairs(train_txt_file, video_dir):
    video_paths = glob.glob(osp.join(video_dir, "*"))

    with open(train_txt_file, "w") as train_txt:
        for video_path in video_paths:
            class_name = str(label_class_dict[osp.basename(video_path)[-12:-8]])
            info = '{:s} {:s}'.format(video_path, class_name)
            train_txt.write(info + '\n')
            print(info)


def generate_video_label_pairs2(train_txt_file, txt_dir):
    txt_paths = glob.glob(osp.join(txt_dir, "*"))
    with open(train_txt_file, 'w') as train_txt:
        for txt_path in txt_paths:
            class_name = str(label_class_dict[osp.basename(txt_path)[-12:-8]])
            video_name = osp.basename(txt_path).split(".")[0] + ".avi"
            info = '{:s} {:s}'.format(video_name, class_name)
            train_txt.write(info + '\n')
            print(info)


if __name__ == "__main__":
    # train_txt_file = "../data/setup_train.txt"
    # txt_dir = "E:/CodeDatasets/Datasets/mutual_cross_setup_detection_tracking_repaired/train"
    # generate_video_label_pairs2(train_txt_file=train_txt_file, txt_dir=txt_dir)

    val_txt_file = "../data/setup_val.txt"
    txt_dir = "E:/CodeDatasets/Datasets/mutual_cross_setup_detection_tracking_repaired/test"
    generate_video_label_pairs2(train_txt_file=val_txt_file, txt_dir=txt_dir)

    """
    train_txt_file = "../data/setup_train.txt"
    # video_dir = "/research/mvg/public/bikechen/mutual_cross_setup/train"
    video_dir = "/scratch/project_2004121/mutual_cross_setup/train"
    generate_video_label_pairs(train_txt_file, video_dir)

    val_txt_file = "../data/setup_val.txt"
    # video_dir = "/research/mvg/public/bikechen/mutual_cross_setup/test"
    video_dir = "/scratch/project_2004121/mutual_cross_setup/test"
    generate_video_label_pairs(val_txt_file, video_dir)
    """

    # train_txt_file = "../data/subject_train.txt"
    # # video_dir = "/research/mvg/public/bikechen/mutual_cross_subject/train"
    # video_dir = "/scratch/project_2004121/mutual_cross_subject/train"
    # generate_video_label_pairs(train_txt_file, video_dir)
    #
    # val_txt_file = "../data/subject_val.txt"
    # # video_dir = "/research/mvg/public/bikechen/mutual_cross_subject/test"
    # video_dir = "/scratch/project_2004121/mutual_cross_subject/test"
    # generate_video_label_pairs(val_txt_file, video_dir)

    print("Successfully!")
