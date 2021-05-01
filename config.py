"""
    Function: Configure parameters for training MutualTimeSformer on the NTU-RGBD mutual interaction dataset.

    Author: Bike Chen
    Email: Bike.Chen@oulu.fi
    Date: April 1, 2021
    Revised: April 25, 2021
"""

import argparse

parser = argparse.ArgumentParser(description="Parameters for training MutualTimeSformer.")

# data constructor.
parser.add_argument('--num_frames', type=int, default=8, help='the number of frames extracted from a video.')
parser.add_argument('--mean', type=tuple, default=(0.5, 0.5, 0.5), help='mean values of the dataset.')
parser.add_argument('--std', type=tuple, default=(0.5, 0.5, 0.5), help='standard values of the dataset.')

parser.add_argument('--train_data_dir_path', type=str, default='data/setup_train.txt',
                    help='a text file showing training video names and corresponding ground truth.')
parser.add_argument('--train_videos_dir', type=str,
                    default='/scratch/project_2004121/mutual_cross_setup/train',
                    # default='/research/mvg/public/bikechen/mutual_cross_setup/train',
                    help='directory of storing training videos.')
parser.add_argument('--train_txt_dir', type=str,
                    default='/scratch/project_2004121/mutual_cross_setup_detection_tracking_together',
                    # default='/research/mvg/public/bikechen/mutual_cross_setup_detection_tracking_together',
                    help='directory of storing bounding boxes and corresponding person ids.')
parser.add_argument('--train_num', type=int, default=11799, help='the number of training samples.')

parser.add_argument('--val_data_dir_path', type=str, default='data/setup_val.txt',
                    help='a text file showing validation video names and corresponding ground truth.')
parser.add_argument('--val_videos_dir', type=str,
                    default='/scratch/project_2004121/mutual_cross_setup/test',
                    # default='/research/mvg/public/bikechen/mutual_cross_setup/test',
                    help='directory of storing validation videos.')
parser.add_argument('--val_txt_dir', type=str,
                    default='/scratch/project_2004121/mutual_cross_setup_detection_tracking_together',
                    # default='/research/mvg/public/bikechen/mutual_cross_setup_detection_tracking_together',
                    help='directory of storing bounding boxes and corresponding person ids.')
parser.add_argument('--val_num', type=int, default=12803, help='the number of validation samples.')

parser.add_argument('--num_classes', type=int, default=26, help='the number of classes.')

# device
parser.add_argument('--gpu_number', type=str, default='0', help='GPU device numbers for training a model.')

# model
parser.add_argument('--dim', type=int, default=256, help='default: 512, embedding size.')
parser.add_argument('--img_size', type=tuple, default=(600, 400), help='image size (height, width).')
parser.add_argument('--patch_size', type=tuple, default=(50, 50), help='patch size (height, width).')
parser.add_argument('--num_frame', type=int, default=8, help='the number of frames extracted from a video')
parser.add_argument('--depth', type=int, default=12, help='the number of multiple head attention layers.')
parser.add_argument('--heads', type=int, default=8, help='the number of heads in Multiple Head Attention.')
parser.add_argument('--dim_head', type=int, default=32, help='head size, sometimes, dim_head x heads = dim.')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='dropout rate in Multiple Head Attention.')
parser.add_argument('--ff_dropout', type=float, default=0.1, help='dropout rate in Fully Connect Layers.')

# train
parser.add_argument('--train_batch_size', type=int, default=32,
                    help='the number of samples used to train the model each time.')
parser.add_argument('--num_workers', type=int, default=24, help='the number of workers used for loading samples.')
parser.add_argument('--val_batch_size', type=int, default=48,
                    help='the number of samples used to evaluate the model each time.')
parser.add_argument('--learning_rate', type=float, default=3e-3, help='the initial learning rate for SGD.')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='hyper-parameter for regularization.')
# parser.add_argument('--num_steps', type=int, default=300, help="total number of training epochs to perform.")
# parser.add_argument('--decay_type', choices=['cosine', 'linear'], default='cosine',
#                     help='how to decay the learning rate.')
# parser.add_argument('--warmup_steps', type=int, default=500,
#                     help='Step of training to perform learning rate warmup for.')
# parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm.')
parser.add_argument('--max_iteration', type=int, default=300, help='the total number of training epochs to perform.')

# save parameters.
parser.add_argument('--save_last_model', type=str, default='checkpoints/saved_last_model.pt',
                    help='including epoch, and model, optimizer, scheduler parameters, and best_val_loss.')
parser.add_argument('--save_best_model', type=str, default='checkpoints/saved_best_model.pt',
                    help='save the best model, only saving parameters.')
parser.add_argument('--save_intime_model', type=str, default='checkpoints/saved_intime_',
                    help='save the in-time model, only saving parameters.')
parser.add_argument('--save_loss_acc', type=str, default='checkpoints/saved_loss_acc.pkl',
                    help='save training loss, accuracy and validation loss, accuracy.')

# continue to train the model if there is a interruption.
parser.add_argument('--is_train', type=bool, default=False, help='Does it train a model?')


if __name__ == "__main__":
    args = vars(parser.parse_args())
    # print(args)
    pretrained_model = args["pretrained_model"]
    print(pretrained_model)


