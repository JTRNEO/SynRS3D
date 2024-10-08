import os
import sys
import argparse
import os.path as osp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
import json
import numpy as np
from evaluation import eval,eval_oem

from utils.utils import adjust_learning_rate

from utils.datasets_config import (
    ss_datasetname, dataset_num_classes, get_dataset_category
)

from torch.utils import data
from ever.core.logger import get_console_file_logger

from dataset.dataset import MultiTaskDataSet, OEMDataSet

from utils.criterion import SmoothL1Loss, CriterionCrossEntropy

from models.dpt import DPT_DINOv2

from torch.utils.tensorboard import SummaryWriter
from albumentations import Compose, RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, OneOf, CenterCrop
from albumentations.pytorch import ToTensorV2
                
def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RS3DAda")

    parser.add_argument("--root_dir", type=str, default='/home/songjian/project/SynRS3D/data/', help="Path to the directory containing the datasets.")
    parser.add_argument("--datasets",  nargs='*', type=str, default=['sr_05_cd_aux'], help="traning dataset name list")
    parser.add_argument("--test_datasets",  nargs='*', type=str, default=['DFC18'], help="target domain 1 datasets list and target domain 2 datasets list")
    parser.add_argument("--ood_datasets",  nargs='*', type=str, default=['DFC18'], help="target domain 2 datasets list")
    parser.add_argument("--images_file", nargs='*', type=str, default=['selected_train.txt', 'test.txt', 'train.txt'], 
                        help="images txt file, first one is the training txt, second is the test txt, third is the style transfer txt")
    parser.add_argument("--crop_size", type=int, default=392, help="height and width of images.")
    parser.add_argument('--decoder', type=str, default='DPT',
                        help='decoder')
    parser.add_argument('--encoder', type=str, default='vits',
                        help='encoder')

    parser.add_argument("--multi_task", action="store_true", help="Whether to add classification branch.")
    parser.add_argument("--combine_class", action="store_true", help="Whether to combine 8 classes to 3.")
    
    parser.add_argument("--apply_da",  nargs='*', type=str, default=['HM', 'PDA'], help="style transfer methods")
    parser.add_argument("--FDA_beta", type=float, default=0.05, help="beta of FDA")
    parser.add_argument("--HM_blend_ratio",  nargs='*', type=float, default=(0.8, 1), help="blend ratio of HM")
    parser.add_argument("--PDA_blend_ratio",  nargs='*', type=float, default=(0.8, 1), help="blend ratio of PDA")
    parser.add_argument("--PDA_type", type=str, default='standard', help="transformation type of PDA")
    parser.add_argument("--tgt_datasets",  nargs='*', type=str, default=['DFC18'], help="target datasets list used for style transfer")
    parser.add_argument("--max_da_images", type=int, default=1200, help="Number of images used for da.")
    
    parser.add_argument("--batch_size", type=int, default=2, help="batchsize")
    
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--decoder_lr_weight", type=float, default=10, help="weight of decoder lr")

    parser.add_argument("--num_steps", type=int, default=40000, help="Number of training steps.")
    parser.add_argument("--start_iters", type=int, default=0, help="start_iters")
    
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=1500, help="Number of warm-up steps.")
    parser.add_argument("--warmup_mode", type=str, default='linear', help="warm-up mode")
    parser.add_argument("--decay_mode", type=str, default='poly', help="decay mode")
    
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=str, default='0', help="choose gpu device.")

    parser.add_argument("--save_num_images", type=int, default=5, help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=500, help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot_dir", type=str, default='snapshot', help="Where to save snapshots of the model.")
    parser.add_argument("--only_save_best", action="store_true", help="only save best")
    
    parser.add_argument("--lambda_dsms", type=float, default=0.8, help="weight of height estimation loss")
    parser.add_argument("--eval_oem", action="store_true", help="eval oem or not")

    parser.add_argument("--pretrained", action="store_true", help="fine-tune the model with large input size.")
    parser.add_argument("--shuffle", action="store_true", help="shuffle or not")
    parser.add_argument("--feat_loss", action="store_true", help="use feat loss or not")
    parser.add_argument("--fl_start", type=int, default=3, help="calculate feature loss from which layer")
    parser.add_argument("--fl_threshold", type=float, default=0.8, help="threshold")
    parser.add_argument("--fl_weight", type=float, default=1., help="threshold")
    parser.add_argument("--fl_decrement", type=float, default=0.05, help="threshold")

    return parser.parse_args()
    
def main():
    args = get_arguments()
    
    args_dict = vars(args)

    # Base directory
    snapshot_dir = args.snapshot_dir

    # Decoder and encoder
    decoder_encoder = f"{args.decoder}_{args.encoder}"
    
    # Determine dataset categories present
    args_datasets = set(args.datasets)

    # Datasets
    datasets = '_'.join(args.datasets)
    
    # Add multi-task or single-task specific paths
    multi_suffixes = []
    if args.multi_task:
        if args.combine_class:
            multi_suffixes.append('multi_task_combine_class')
        else:
            multi_suffixes.append('multi_task_ori_class')
    else:
        multi_suffixes.append('single_task')
    
    multi_suffix_combined = '_'.join(multi_suffixes)
    # Combining all parts
    SNAPSHOT_DIR = os.path.join(snapshot_dir, 
                                decoder_encoder, 
                                f"{args.crop_size}"+'_'+f"lr_{args.learning_rate}"+'_'+f"wd_{args.weight_decay}",
                                multi_suffix_combined,
                                datasets)

    """Create the model and start the training."""
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)

    config_file_path = os.path.join(SNAPSHOT_DIR, 'config.json')
    
    # Convert args to a JSON string and write to the file
    with open(config_file_path, 'w') as config_file:
        json.dump(args_dict, config_file, indent=4)

    logger = get_console_file_logger(name=args.decoder + '_' + args.encoder, level=logging.INFO, logdir=SNAPSHOT_DIR)
    
    writer = SummaryWriter(log_dir=SNAPSHOT_DIR + '/runs')

    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ss_num_classes = 3
    train_dataset_type = get_dataset_category(args_datasets)
    
    if args.multi_task:
        # Once we've validated the conditions, determine the number of classes based on combine_class flag
        if args.combine_class:
            ss_num_classes = 3
        else:
            # This case handles non-combine class scenarios not specific to mix
            if not args_datasets.issubset(ss_datasetname):
                raise ValueError('So far, multi-task training only supports datasets with ss labels.')
            ss_num_classes = dataset_num_classes[train_dataset_type]

    regression_config = [
        {
            'name': 'regression',
            'nclass': 1,  # Number of classes for the segmentation mask of the first task
        }]


    segmentation_config = [
        {
            'name': 'segmentation',
            'nclass': ss_num_classes  # Number of classes for the segmentation mask of the first task
        }]
    
    cudnn.enabled = True
    # -----------------------------
    # Create network.
    # -----------------------------
    if args.multi_task:
        head_configs = regression_config + segmentation_config
    else:
        head_configs = regression_config

    model = DPT_DINOv2(encoder=args.encoder, head_configs=head_configs, pretrained=args.pretrained)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if args.feat_loss:
        target_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(args.encoder), pretrained=args.pretrained)        
        target_encoder.cuda()
        for param in target_encoder.parameters():
            param.requires_grad = False
    
    height_criterions = SmoothL1Loss(reduction='none')
    cosine_similarity = torch.nn.CosineSimilarity(dim=-1)    
    
    if args.multi_task:
        ss_criterion = CriterionCrossEntropy(reduction='none', ignore_index=255)
    
    cudnn.benchmark = True
    
    da_aug_paras = {'FDA': {'beta_limit': args.FDA_beta},
                    'HM': {'blend_ratio': args.HM_blend_ratio},
                    'PDA': {'blend_ratio': args.PDA_blend_ratio, 'transform_type': args.PDA_type}}
    

    traning_src_transforms = Compose([
    RandomCrop(args.crop_size, args.crop_size),
    OneOf([
        HorizontalFlip(True),
        VerticalFlip(True),
        RandomRotate90(True)
    ], p=0.75),
    Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
    ToTensorV2()
    ])

    tgt_data_path = [os.path.join(args.root_dir, dataset) for dataset in args.tgt_datasets]
        
    # Filter out real dataset names from args.datasets for real_train_data_path
    # Filter out synthetic dataset names (from all synthetic datasets) for syn_train_data_path
    syn_train_data_path = [os.path.join(args.root_dir, dataset) for dataset in args.datasets]
    
    tgt_data_path = [os.path.join(args.root_dir, dataset) for dataset in args.tgt_datasets]
    
    syn_traindataset = MultiTaskDataSet(syn_train_data_path, 
                                is_training=True, 
                                images_file=args.images_file,
                                transforms=traning_src_transforms,
                                max_iters=args.num_steps * args.batch_size,
                                max_da_images=args.max_da_images,
                                multi_task=args.multi_task,
                                combine_class=args.combine_class,
                                apply_da=args.apply_da,
                                da_aug_paras=da_aug_paras,
                                tgt_root_dir = tgt_data_path
                                )
    syn_trainloader = data.DataLoader(syn_traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    test_data_path = [os.path.join(args.root_dir, dataset) for dataset in args.test_datasets]
    testing_transforms = Compose([
    CenterCrop(504, 504),
    Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
    ToTensorV2()
    ])

    testloaders = {}
    for d in test_data_path:
        base_folder_name = os.path.basename(d)  # Extract the base folder name
        testdataset = MultiTaskDataSet([d], 
                                is_training=False,
                                images_file=args.images_file,
                                transforms=testing_transforms,
                                multi_task=True if args.multi_task and base_folder_name in ss_datasetname else False,
                                combine_class=False if not args.combine_class and get_dataset_category(set([base_folder_name]))==train_dataset_type else True,
                                )
        testloader = data.DataLoader(testdataset, batch_size=1, shuffle=False)
        testloaders[base_folder_name] = testloader  # Store using the base folder name as the key
    
    if args.eval_oem: 
        oemloaders = {}
        oemdataset = OEMDataSet([os.path.join(args.root_dir,'OEM')], 
                                is_training=False,
                                images_file=args.images_file,
                                transforms=testing_transforms,
                                combine_class=False if not args.combine_class and get_dataset_category(set(['OEM']))==train_dataset_type else True, 
                                )
        oemloader = data.DataLoader(oemdataset, batch_size=1, shuffle=False)
        oemloaders['OEM']=oemloader

    encoder_modules = set(model.pretrained.parameters())
    all_modules = set(model.parameters())
    decoder_modules = all_modules-encoder_modules
    # Specify parameter groups
    encoder_params = {
        'params': list(encoder_modules),  # corrected 'ecoder' to 'encoder'
        'lr': args.learning_rate,
        'weight_decay': args.weight_decay,
        'init_lr': args.learning_rate,
        'name': 'encoder'
    }

    decoder_params = {
        'params': list(decoder_modules),
        'lr': args.learning_rate * args.decoder_lr_weight,  # assuming 'decoder_lr_weight' is the correct name
        'weight_decay': args.weight_decay,
        'init_lr': args.learning_rate * args.decoder_lr_weight,
        'name': 'decoder'
    }

    # Creating the optimizer with specific parameter groups
    optimizer = optim.AdamW([encoder_params, decoder_params])
    
    optimizer.zero_grad()
    best_metrics = {'HE': float('inf'), 'SS': float('-inf')}
    best_model_paths = {'HE': None, 'SS': None}
    
    for i_iter in range(args.start_iters, args.num_steps):
        # training on source
        batch = next(iter(syn_trainloader))
        images, dsms = batch['image'], batch['dsm']
        ss_masks = batch.get('ss_mask') if args.multi_task else None

        # Move tensors to GPU and handle data types
        images, dsms = images.cuda(), dsms.cuda()
        if ss_masks is not None:
            ss_masks = ss_masks.squeeze(dim=1).long().cuda()

        model.train()
        optimizer.zero_grad()
        
        lr = adjust_learning_rate(optimizer,args.learning_rate, i_iter, args.num_steps, args.power, args.warmup_steps, args.warmup_mode, args.decay_mode)
        if args.feat_loss:
            # Define base threshold and how much it decreases with each layer
            base_threshold = args.fl_threshold  # Assume this is the initial threshold for the last layer (layer 5)
            threshold_decrement = args.fl_decrement  # This value determines how much the threshold decreases per layer
            total_fl_loss = torch.tensor(0.0, device=model.parameters().__next__().device)
            # Loop over layers 1 to 5
            for layer_index in range(args.fl_start, 4):
                pre_feat = model.pretrained.get_intermediate_layers(images, 4, return_class_token=True)[layer_index][0]  # Extract features from current layer
                target_feat = target_encoder.get_intermediate_layers(images, 4, return_class_token=True)[layer_index][0]  # Extract target features from current layer
                cos_sim = cosine_similarity(pre_feat, target_feat)
                # Decrease the threshold as the layer index increases
                current_threshold = base_threshold - (threshold_decrement * layer_index)

                # Create a mask based on the current threshold
                mask = cos_sim < current_threshold

                # Apply the mask - only compute loss where mask is True
                selected_cos_sim = torch.masked_select(cos_sim, mask)
                if selected_cos_sim.numel() > 0:  # Check if there are any elements below threshold
                    feat_loss_cosine = (1 - selected_cos_sim.mean())
                    # Accumulate loss for each layer, assuming 'total_loss' is defined outside the loop
                    total_fl_loss += feat_loss_cosine
                    
            total_fl_loss *= args.fl_weight
            
        pre_outputs = model(images)

        pre_dsms = pre_outputs.get('regression', None)
        pre_ss_masks = pre_outputs.get('segmentation', None) if args.multi_task else None

        total_loss = torch.tensor(0.0, device=model.parameters().__next__().device)
        loss_dict = {}
        
        src_he_loss = (height_criterions(pre_dsms, dsms).mean())*args.lambda_dsms
        loss_dict["height_loss"] = src_he_loss.item()  # Keep track of individual losses if needed
        total_loss += src_he_loss
        
        if args.multi_task:
            src_seg_loss = ss_criterion(pre_ss_masks, ss_masks).mean()
            loss_dict['segmentation_loss'] = src_seg_loss.item()
            total_loss += src_seg_loss
        if args.feat_loss:
            loss_dict['feat_loss'] = total_fl_loss.item()
            total_loss += total_fl_loss      
                
        # Backpropagation
        total_loss.backward()
        optimizer.step()

        if i_iter % 100 == 0:
            # Assuming 'total_loss' and individual losses in 'loss_dict' are already calculated as shown previously
            # Convert total_loss to a numpy value for logging
            full_loss_value = total_loss.item()  # For PyTorch >= 0.4.1, .item() is preferred for single element tensors

            # Prepare dictionary for easy logging of all loss components
            loss_values = {loss_name: value for loss_name, value in loss_dict.items()}

            # Assuming each param group could be identified by a name (you might need to add 'name' keys when setting up param groups)
            current_lrs = {pg.get('name', 'Group_{}'.format(i)): pg['lr'] for i, pg in enumerate(optimizer.param_groups)}

            # Log values with named learning rates
            logger.info('[Train on {} model]: iter:{}/{} | Full Loss = {} | {} | LRs = {}'.format(
                args.decoder + '_' + args.encoder, i_iter, args.num_steps,
                full_loss_value, ' | '.join([f"{k} = {v:.4f}" for k, v in loss_values.items()]), current_lrs))

            # Record values in TensorBoard
            writer.add_scalar('Loss/train', full_loss_value, i_iter)
            for loss_name, loss_value in loss_values.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value, i_iter)
            # Assume each parameter group has a 'name' key for identification (as set up in previous examples)
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = param_group.get('name', f'param_group_{i}')
                writer.add_scalar(f'Learning Rate/{group_name}', param_group['lr'], i_iter)

            
        if i_iter!= 0 and i_iter % args.save_pred_every == 0:
            if args.eval_oem:
                eval_oem(oemloaders, model, args.save_num_images, writer, logger, i_iter, args=args, train_dataset_type=train_dataset_type)
                
            results = eval(testloaders, model, args.save_num_images, writer, logger, i_iter, args=args, train_dataset_type=train_dataset_type)
            # Initialize new best flags for each metric
            new_best_HE = False
            new_best_SS = False

            if results['HE'] < best_metrics['HE']:
                best_metrics['HE'] = results['HE']
                new_best_HE = True
                logger.info(f"New best HE: {results['HE']} at iteration {i_iter}")

            if 'SS' in results and results['SS'] > best_metrics['SS']:
                best_metrics['SS'] = results['SS']
                new_best_SS = True
                logger.info(f"New best SS: {results['SS']} at iteration {i_iter}")

            # Save and possibly delete old best HE model
            if new_best_HE and args.only_save_best:
                old_he_path = best_model_paths['HE']
                new_he_path = osp.join(SNAPSHOT_DIR, f"best_HE_model_{i_iter}.pth")
                best_model_paths['HE'] = new_he_path
                torch.save(model.state_dict(), new_he_path)
                if old_he_path is not None and os.path.exists(old_he_path):
                    os.remove(old_he_path)

                logger.info(f"Saved new best HE model at iteration {i_iter}")

            # Save and possibly delete old best SS model
            if new_best_SS and args.only_save_best:
                old_ss_path = best_model_paths['SS']
                new_ss_path = osp.join(SNAPSHOT_DIR, f"best_SS_model_{i_iter}.pth")
                best_model_paths['SS'] = new_ss_path
                torch.save(model.state_dict(), new_ss_path)
                if old_ss_path is not None and os.path.exists(old_ss_path):
                    os.remove(old_ss_path)

                logger.info(f"Saved new best SS model at iteration {i_iter}")

            # Save regular checkpoint if not only saving best models
            if not args.only_save_best:
                regular_checkpoint_path = osp.join(SNAPSHOT_DIR, f"{i_iter}.pth")
                torch.save(model.state_dict(), regular_checkpoint_path)
                logger.info(f"Saved regular model checkpoint at iteration {i_iter}")

if __name__ == '__main__':
    main()