"""
    Function: test trained MutualTimeSformer model on the mutual interaction dataset (extracted from NTU-GRBD-120).

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 29, 2021
"""

import random
import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.utils.data as Data

from utils.data_constructor import DatasetConstructor
from utils.tools import compute_accuracy
from models.mutual_timesformer import MutualTimeSformer
from config import parser


if __name__ == "__main__":
    args = vars(parser.parse_args())

    # fix all random seeds
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.deterministic = True
    #  torch.backends.cudnn.benchmark = True

    # dataset.
    val_data_dir_path = args["val_data_dir_path"]
    val_videos_dir = args['val_videos_dir']
    val_txt_dir = args['val_txt_dir']
    val_num = args["val_num"]

    val_dataset = DatasetConstructor(args=args,
                                     data_dir_path=val_data_dir_path,
                                     videos_dir=val_videos_dir,
                                     txt_dir=val_txt_dir,
                                     num=val_num,
                                     is_train=False)

    val_loader = Data.DataLoader(dataset=val_dataset,
                                 batch_size=args["val_batch_size"],
                                 num_workers=args['num_workers'],
                                 pin_memory=True)

    # device.
    device = torch.device("cuda:"+args["gpu_number"] if torch.cuda.is_available() else "cpu")

    # model.
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
    # should only load parameters not including the model architecture.
    saved_dict = torch.load(args["save_last_model"])
    net.load_state_dict(saved_dict['net'])  # load pretrained parameters.
    net.to(device)
    net.eval()

    # criteria
    criteria = nn.CrossEntropyLoss()

    # iteration.
    val_loss = []
    val_acc = []

    # validation
    losses = []
    all_preds = []
    all_gt = []

    progressbar = tqdm.tqdm(range(len(val_loader)))

    for index, val_video, val_gt, ti in val_loader:
        val_video = val_video.to(device)
        val_gt = val_gt.to(device)

        with torch.no_grad():
            logits = net(val_video)

            # cross entropy loss
            loss = criteria(logits, val_gt)
            losses.append(loss.detach().item())

            # Compute Accuracy
            preds = torch.argmax(logits, dim=-1)
            if len(all_preds) == 0:
                all_preds.append(preds.detach().cpu().numpy())
                all_gt.append(val_gt.detach().cpu().numpy())
            else:
                all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
                all_gt[0] = np.append(all_gt[0], val_gt.detach().cpu().numpy(), axis=0)

        progressbar.set_description("loss {0:.2f}\n".format(loss.detach().item()))
        progressbar.update(1)

    progressbar.close()
    val_loss.append(np.mean(losses))
    val_acc = compute_accuracy(all_preds[0], all_gt[0])
    print("Val loss {0:.2f}, Val acc {1:.2f}".format(val_loss[-1], val_acc))
    print("Test the TimeSformer successfully.")
