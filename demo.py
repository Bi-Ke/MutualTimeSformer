"""
    Function: test pretrained MutualTimeSformer model.

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 29, 2021
"""
import torch
import os.path as osp

from config import parser
from utils.data_constructor import ReadVideo
from utils.generate_video_label_pairs import label_name_dict, label_class_dict
from models.mutual_timesformer import MutualTimeSformer


args = vars(parser.parse_args())
device = torch.device("cuda:"+args["gpu_number"] if torch.cuda.is_available() else "cpu")

# prepare data.
video_name = "S002C001P003R001A050_rgb.avi"  # S002C001P003R001A051_rgb.avi
# video_name = "S002C001P003R001A051_rgb.avi"
videos_dir = "samples"
txt_dir = "samples"

read_video = ReadVideo(args)
video_tensor = read_video.extract_video_frames(video_name, videos_dir, txt_dir)
video_tensor = video_tensor.unsqueeze(dim=0).to(device)

# load the trained MutualTimeSformer.
net = MutualTimeSformer(
    dim=args['dim'],  # 256
    image_size=args['img_size'],  # (600, 400)
    patch_size=args['patch_size'],  # (50, 50)
    num_frames=args['num_frames'],  # 8
    num_classes=args['num_classes'],  # 26
    depth=args['depth'],  # 12
    heads=args['heads'],  # 8
    dim_head=args['dim_head'],  # 32, theoretically, dim_head * heads = dim.
    attn_dropout=args['attn_dropout'],  # 0.1,
    ff_dropout=args['ff_dropout']  # 0.1
)

saved_dict = torch.load(args["save_last_model"])
net.load_state_dict(saved_dict['net'])  # load pretrained parameters.
net.to(device)
net.eval()

# start_epoch = saved_dict['epoch']  # 32, 2021.4.29, 9:07
# print(start_epoch)

# get the predicted result.
with torch.no_grad():
    logits = net(video_tensor)
    preds = torch.argmax(logits, dim=-1)

pred = int(preds[0].cpu().numpy())

# compare predicted results with ground truth.
gt = label_class_dict[video_name[-12:-8]]
gt_class_name = label_name_dict[video_name[-12:-8]]

class_label_dict = {v: k for k, v in label_class_dict.items()}
label_name = class_label_dict[pred]
pred_class_name = label_name_dict[label_name]

print(pred_class_name)
print(gt_class_name)

