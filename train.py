"""
    Function: training or fine-tuning the MutualTimeSformer model.

    Please note that the code can also be used to continue to train the model if there is an interruption.

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 3, 2021
    Revised: April 25, 2021
"""

import random
import numpy as np
import tqdm
import pickle
import time

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
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

    # dataset. data_dir_path, videos_dir, txt_dir, num,
    train_data_dir_path = args["train_data_dir_path"]
    train_videos_dir = args['train_videos_dir']
    train_txt_dir = args['train_txt_dir']
    train_num = args["train_num"]

    val_data_dir_path = args["val_data_dir_path"]
    val_videos_dir = args['val_videos_dir']
    val_txt_dir = args['val_txt_dir']
    val_num = args["val_num"]

    train_dataset = DatasetConstructor(args=args,
                                       data_dir_path=train_data_dir_path,
                                       videos_dir=train_videos_dir,
                                       txt_dir=train_txt_dir,
                                       num=train_num,
                                       is_train=True)

    val_dataset = DatasetConstructor(args=args,
                                     data_dir_path=val_data_dir_path,
                                     videos_dir=val_videos_dir,
                                     txt_dir=val_txt_dir,
                                     num=val_num,
                                     is_train=False)

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   batch_size=args["train_batch_size"],
                                   num_workers=args['num_workers'],
                                   pin_memory=True)

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
    if not args['is_train']:
        saved_dict = torch.load(args["save_last_model"])
        net.load_state_dict(saved_dict['net'])  # load pretrained parameters.
    net.to(device)

    # optimization.
    parameters = net.parameters()
    # optimizer = optim.SGD(parameters,
    #                       lr=args["learning_rate"],
    #                       momentum=0.9,
    #                       weight_decay=args['weight_decay'])

    optimizer = optim.Adam(parameters,
                           lr=args["learning_rate"],
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=args['weight_decay'])

    if not args['is_train']:
        optimizer.load_state_dict(saved_dict['optim'])  # load pretrained parameters.

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 3e-3

    max_iteration = args['max_iteration']
    # if args['decay_type'] == 'cosine':
    #     scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=max_iteration)
    # else:
    #     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args['warmup_steps'], t_total=max_iteration)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min',
                                                     factor=0.1,
                                                     patience=90,
                                                     verbose=True)

    if not args['is_train']:
        scheduler.load_state_dict(saved_dict['scheduler'])  # load pretrained parameters.

    # criteria
    criteria = nn.CrossEntropyLoss()

    # iteration.
    train_loss = []
    # val_loss = []
    # val_acc = []

    # select the best model based on the overall loss.
    best_val_loss = np.inf
    if not args['is_train']:
        best_val_loss = saved_dict['best_val_loss']

    start_epoch = 0
    if not args['is_train']:
        start_epoch = saved_dict['epoch']

    for epoch in range(start_epoch, max_iteration):
        print(f'epoch : {epoch}')
        losses = []
        # train
        net.train()

        train_dataset = train_dataset.shuffle()
        progressbar = tqdm.tqdm(range(len(train_loader)))

        step = 0
        for index, train_video, train_gt, ti in train_loader:
            train_video = train_video.to(device)
            train_gt = train_gt.to(device)

            optimizer.zero_grad()
            logits = net(train_video)

            # Cross Entropy Loss
            loss = criteria(logits, train_gt)
            losses.append(loss.detach().item())

            loss.backward()
            optimizer.step()
            # scheduler.step()

            progressbar.set_description("epoch {0:.2f}, loss {1:.2f}       \n".format(epoch, loss.detach().item()))
            progressbar.update(1)

        train_loss.append(np.mean(losses))
        progressbar.close()

        # only save the last model.
        save_dict = {'epoch': epoch,
                     'net': net.state_dict(),
                     'optim': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(),
                     'best_val_loss': best_val_loss}

        torch.save(save_dict, args['save_last_model'])
        print('model is saved: {:s}'.format(args['save_last_model']))

        # del loss, losses
        # torch.cuda.empty_cache()

        # validation
        # net.eval()
        # losses = []
        # all_preds = []
        # all_gt = []
        #
        # progressbar = tqdm.tqdm(range(len(val_loader)))
        # for index, val_video, val_gt, ti in val_loader:
        #     val_video = val_video.to(device)
        #     val_gt = val_gt.to(device)
        #
        #     with torch.no_grad():
        #         logits = net(val_video)
        #
        #         # cross entropy loss
        #         loss = criteria(logits, val_gt)
        #         losses.append(loss.detach().item())
        #
        #         # Compute Accuracy
        #         preds = torch.argmax(logits, dim=-1)
        #         if len(all_preds) == 0:
        #             all_preds.append(preds.detach().cpu().numpy())
        #             all_gt.append(val_gt.detach().cpu().numpy())
        #         else:
        #             all_preds[0] = np.append(all_preds[0], preds.detach().cpu().numpy(), axis=0)
        #             all_gt[0] = np.append(all_gt[0], val_gt.detach().cpu().numpy(), axis=0)
        #
        #     progressbar.set_description("epoch {0:.2f}, loss {1:.2f}\n".format(epoch, loss.detach().item()))
        #     progressbar.update(1)
        #
        # progressbar.close()
        # val_loss.append(np.mean(losses))
        # val_acc = compute_accuracy(all_preds[0], all_gt[0])
        # print("Train loss {0:.2f}, Val loss {1:.2f}, Val acc {2:.2f}".
        #       format(train_loss[-1], val_loss[-1], val_acc))

        scheduler.step(train_loss[-1])

        # if val_loss[-1] < best_val_loss:
        #     best_val_loss = val_loss[-1]
        #     print("Best Model !")
        #     torch.save(net.state_dict(), args["save_best_model"])
        #
        # if (epoch + 1) % 50 == 0:
        #     torch.save(net.state_dict(), args["save_intime_model"]+str(epoch+1)+".pt")
        #
        # del loss, losses, all_preds, all_gt
        # torch.cuda.empty_cache()

        # with open(args['save_loss_acc'], 'ab') as file:
        #     pickle.dump({'train_loss': train_loss[-1],
        #                  'val_loss': val_loss[-1],
        #                  'val_acc': val_acc}, file)

        with open(args['save_loss_acc'], 'ab') as file:
            pickle.dump({'train_loss': train_loss[-1]}, file)

        print(time.strftime('%H:%M:%S', time.localtime()))

    print("Training the MutualTimeSformer successfully.")

