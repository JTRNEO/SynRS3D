import os
import sys
import argparse
import os.path as osp
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import logging
from torchvision.utils import make_grid
import json
import numpy as np

from utils.utils import print_metrics

from utils.datasets_config import (
    ss_datasetname, dataset_num_classes, get_dataset_category
)
from tqdm import tqdm
from dataset.dataset import MultiTaskDataSet, PesudoDataSet, OEMDataSet, labelmap
from torch.utils import data
from torch.nn import functional as F
from ever.core.logger import get_console_file_logger
import ever.api.metric as ss_metric

from dataset.dataset import labelmap

from utils.metrics import AverageMeter, Result

from models.dpt import DPT_DINOv2

from torch.utils.tensorboard import SummaryWriter
from utils.vis import convert_dsm, convert_ss_mask, prepare_image_for_tensorboard
from albumentations import Compose, Normalize, CenterCrop
from albumentations.pytorch import ToTensorV2


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--restore_path", type=str, default='', help="trained model path")
    parser.add_argument("--num_classes", type=int, default=8, help="#classes of land cover branch")
    parser.add_argument("--eval_oem", action="store_true", help="eval oem or not")
    parser.add_argument("--test_datasets",  nargs='*', type=str, default=['DFC18'], help="data name list")
    parser.add_argument("--ood_datasets",  nargs='*', type=str, default=['DFC18'], help="data name list")
    parser.add_argument("--images_file", nargs='*', type=str, default=['train.txt', 'test_syn.txt', 'test.txt'], help="images txt file")

    parser.add_argument("--save_num_images", type=int, default=5, help="How many images to save.")
    parser.add_argument("--snapshot_dir", type=str, default='snapshot', help="Where to save snapshots of the model.")

    return parser.parse_args()
    
def main():
    args = get_arguments()
    """Create the model and start the training."""
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    logger = get_console_file_logger(name=args.decoder + '_' + args.encoder, level=logging.INFO, logdir=args.snapshot_dir)
    
    writer = SummaryWriter(log_dir=args.snapshot_dir + '/runs')
    
    args_datasets = set(args.datasets)
    train_dataset_type = get_dataset_category(args_datasets)

    if not args.gpu == 'None':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    regression_config = [
        {
            'name': 'regression',
            'nclass': 1,  # Number of classes for the segmentation mask of the first task
        }]


    segmentation_config = [
        {
            'name': 'segmentation',
            'nclass': args.num_classes  # Number of classes for the segmentation mask of the first task
        }]
    
    cudnn.enabled = True
    # -----------------------------
    # Create network.
    # -----------------------------
    head_configs = regression_config + segmentation_config
        
    model = DPT_DINOv2(encoder=args.encoder, head_configs=head_configs, pretrained=args.pretrained)

    saved_state_dict = torch.load(args.restore_path)
    model.load_state_dict(saved_state_dict)
    print("model loaded")
    model.cuda()
    
    cudnn.benchmark = True
    
    test_data_path = [os.path.join(args.root_dir, dataset) for dataset in args.test_datasets]
    testing_transforms = Compose([
    CenterCrop(504, 504),        
    Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
    ToTensorV2()
    ])
   
    oemloaders = {}
    oemdataset = OEMDataSet([os.path.join(args.root_dir,'OEM')], 
                            is_training=False,
                            images_file=args.images_file,
                            transforms=testing_transforms,
                            combine_class=False if not args.combine_class and get_dataset_category(set(['OEM']))==train_dataset_type else True, 
                            )
    oemloader = data.DataLoader(oemdataset, batch_size=1, shuffle=False)
    oemloaders['OEM']=oemloader

    testloaders = {}
    for d in test_data_path:
        base_folder_name = os.path.basename(d)  # Extract the base folder name
        testdataset = MultiTaskDataSet([d], 
                                is_training=False,
                                images_file=args.images_file,
                                transforms=testing_transforms,
                                multi_task=True if args.multi_task and base_folder_name in ss_datasetname else False,
                                combine_class=False if not args.combine_class and get_dataset_category(set([base_folder_name]))==train_dataset_type else True, 
                                r=args.segment_r,
                                even_0_3=args.even_0_3,
                                even_3_b=args.even_3_b,
                                )
        testloader = data.DataLoader(testdataset, batch_size=1, shuffle=False)
        testloaders[base_folder_name] = testloader  # Store using the base folder name as the key
        
    if args.eval_oem:
        eval_oem(oemloaders, model, args.save_num_images, writer, logger, 0, args=args, train_dataset_type=train_dataset_type)
        
    eval(testloaders, model, args.save_num_images, writer, logger, 0, args=args,train_dataset_type=train_dataset_type)
            
def eval_oem(testloaders, model, num_images_to_save, writer, logger, i_iter, args=None, train_dataset_type=None):
    
    eval_combination_relabel_rules = {
    "OEM": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 2},
    "DFC19": {0: 0, 1: 1, 2: 2, 3: 0, 4: 0},
    "ISPRS": {0: 0, 1: 2, 2: 0, 3: 1, 4: 0, 5: 0}
    }
    
    model.eval()
    
    for dataset_name, testloader in testloaders.items():
        logger.info(f"Testing on dataset: {dataset_name}")

        vis_results = {'ori_img': [], 'tb_gt_ss_masks':[], 'tb_pre_ss_masks': []}
        
        eval_combine_class = not (not args.combine_class and (get_dataset_category({dataset_name}) in [train_dataset_type, 'OEM']))

        if eval_combine_class:
            ss_metric_op = ss_metric.PixelMetric(num_classes = 3, logger=logger)
        else:
            ss_metric_op = ss_metric.PixelMetric(num_classes = dataset_num_classes[get_dataset_category(set([dataset_name]))])
        
        for index, batch in enumerate(tqdm(testloader)):
            images = batch['image']
            ss_masks = batch.get('ss_mask', None)

            ss_masks = np.squeeze(ss_masks.cpu().numpy().astype(np.uint8))
            mask = (ss_masks >= 0) & (ss_masks != 255)
            
            with torch.no_grad():
                pre_outputs = model(images.cuda())

                pre_ss_logits = pre_outputs['segmentation']
                pre_ss_masks = np.squeeze(np.argmax(pre_ss_logits.cpu().numpy(), axis=1))
                if eval_combine_class != args.combine_class:
                    pre_ss_masks = labelmap(pre_ss_masks, eval_combination_relabel_rules[train_dataset_type])
                y_true = ss_masks[mask].ravel()
                y_pred = pre_ss_masks[mask].ravel()

                ss_metric_op.forward(y_true, y_pred)
            # Check if the current image is one of the ones we want to save
            if index < num_images_to_save:
                ori_img = prepare_image_for_tensorboard(images)
                vis_results['ori_img'].append(ori_img)

                tb_gt_ss_masks = convert_ss_mask(ss_masks)
                tb_pre_ss_masks =convert_ss_mask(pre_ss_masks)
                vis_results['tb_pre_ss_masks'].append(tb_pre_ss_masks)
                vis_results['tb_gt_ss_masks'].append(tb_gt_ss_masks)

        if vis_results['tb_gt_ss_masks']:
            gt_grid = make_grid(vis_results['tb_gt_ss_masks'], nrow=len(vis_results['tb_gt_ss_masks']))
            pred_grid = make_grid(vis_results['tb_pre_ss_masks'], nrow=len(vis_results['tb_pre_ss_masks']))
            writer.add_image(f'{dataset_name}_Gt_SS_Grid', gt_grid, i_iter)
            writer.add_image(f'{dataset_name}_Pre_SS_Grid', pred_grid, i_iter)
            

        avg_ss_metric = ss_metric_op.summary_all()
        writer.add_scalar('{}/acc'.format(dataset_name), avg_ss_metric.rows[-2][1], i_iter)
        writer.add_scalar('{}/mean_iu'.format(dataset_name), avg_ss_metric.rows[-3][1], i_iter)
        for k,v in enumerate(avg_ss_metric.rows[:-3]):
            writer.add_scalar('{}/class_{}_iu'.format(dataset_name, k), v[1], i_iter)

def eval(testloaders, model, num_images_to_save, writer, logger, i_iter, args=None, train_dataset_type=None):
    
    eval_combination_relabel_rules = {
    "OEM": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 2},
    "DFC19": {0: 0, 1: 1, 2: 2, 3: 0, 4: 0},
    "ISPRS": {0: 0, 1: 2, 2: 0, 3: 1, 4: 0, 5: 0}
    }
    
    model.eval()
    # Overall average meters for all datasets
    overall_average_meter = AverageMeter()
    overall_ood_average_meter = AverageMeter()
    total_iou = 0
    number_ss_data = 0
        
    for dataset_name, testloader in testloaders.items():
        logger.info(f"Testing on dataset: {dataset_name}")
        average_meter = AverageMeter()
        vis_results = {'ori_img': [], 'tb_gt_dsms':[], 'tb_pre_dsms': [], 'tb_gt_ss_masks':[], 'tb_pre_ss_masks': []}
        
        eval_multi_task=True if args.multi_task and dataset_name in ss_datasetname else False
        eval_combine_class=False if not args.combine_class and get_dataset_category(set([dataset_name])) == train_dataset_type else True
        
        if eval_multi_task:
            if eval_combine_class:
                ss_metric_op = ss_metric.PixelMetric(num_classes = 3, logger=logger)
            else:
                ss_metric_op = ss_metric.PixelMetric(num_classes = dataset_num_classes[get_dataset_category(set([dataset_name]))])
        
        for index, batch in enumerate(tqdm(testloader)):
            images, dsms = batch['image'], batch['dsm']
            ss_masks = batch.get('ss_mask', None)
            if ss_masks is not None and eval_multi_task:
                ss_masks = np.squeeze(ss_masks.cpu().numpy().astype(np.uint8))
                mask = ss_masks >= 0
            with torch.no_grad():
                pre_outputs = model(images.cuda())
                pre_dsms = pre_outputs.get('regression', None).cpu().numpy()
                if ss_masks is not None and eval_multi_task:
                    pre_ss_logits = pre_outputs['segmentation']
                    pre_ss_masks = np.squeeze(np.argmax(pre_ss_logits.cpu().numpy(), axis=1))
                    if eval_combine_class != args.combine_class:
                        pre_ss_masks = labelmap(pre_ss_masks, eval_combination_relabel_rules[train_dataset_type])
                    y_true = ss_masks[mask].ravel()
                    y_pred = pre_ss_masks[mask].ravel()

            pre_dsms = np.squeeze(pre_dsms)
            dsms = np.squeeze(dsms.cpu().numpy())

            if ss_masks is not None and eval_multi_task:
                ss_metric_op.forward(y_true, y_pred)
            
            result = Result()
            result.update(pre_dsms, dsms)
                
            average_meter.update(result, images.size(0))
            # Check if the current image is one of the ones we want to save
            if index < num_images_to_save:
                tb_gt_dsms, max_value = convert_dsm(dsms)
                tb_pre_dsms, _ = convert_dsm(pre_dsms, max_value=max_value)
                ori_img = prepare_image_for_tensorboard(images)
                vis_results['ori_img'].append(ori_img)
                vis_results['tb_pre_dsms'].append(tb_pre_dsms)
                vis_results['tb_gt_dsms'].append(tb_gt_dsms)
                if ss_masks is not None and eval_multi_task:
                    tb_gt_ss_masks = convert_ss_mask(ss_masks)
                    tb_pre_ss_masks =convert_ss_mask(pre_ss_masks)
                    vis_results['tb_pre_ss_masks'].append(tb_pre_ss_masks)
                    vis_results['tb_gt_ss_masks'].append(tb_gt_ss_masks)

        if vis_results['tb_gt_dsms']:
            ori_img_grid = make_grid(vis_results['ori_img'], nrow=len(vis_results['ori_img']))
            gt_grid = make_grid(vis_results['tb_gt_dsms'], nrow=len(vis_results['tb_gt_dsms']))
            pred_grid = make_grid(vis_results['tb_pre_dsms'], nrow=len(vis_results['tb_pre_dsms']))
            writer.add_image(f'{dataset_name}_Opt_Grid', ori_img_grid, i_iter)
            writer.add_image(f'{dataset_name}_Gt_nDSM_Grid', gt_grid, i_iter)
            writer.add_image(f'{dataset_name}_Pre_nDSM_Grid', pred_grid, i_iter)
        if vis_results['tb_gt_ss_masks']:
            gt_grid = make_grid(vis_results['tb_gt_ss_masks'], nrow=len(vis_results['tb_gt_ss_masks']))
            pred_grid = make_grid(vis_results['tb_pre_ss_masks'], nrow=len(vis_results['tb_pre_ss_masks']))
            writer.add_image(f'{dataset_name}_Gt_SS_Grid', gt_grid, i_iter)
            writer.add_image(f'{dataset_name}_Pre_SS_Grid', pred_grid, i_iter)
            
        if ss_masks is not None and eval_multi_task:    
            avg_ss_metric = ss_metric_op.summary_all()
            total_iou += avg_ss_metric.rows[-3][1]
            number_ss_data += 1
            writer.add_scalar('{}/acc'.format(dataset_name), avg_ss_metric.rows[-2][1], i_iter)
            writer.add_scalar('{}/mean_iu'.format(dataset_name), avg_ss_metric.rows[-3][1], i_iter)
            for k,v in enumerate(avg_ss_metric.rows[:-3]):
                writer.add_scalar('{}/class_{}_iu'.format(dataset_name, k), v[1], i_iter)
                
        avg_metrics = average_meter.average()
        print_metrics(avg_metrics, logger, dataset_name)
        
        overall_average_meter.aggregate(avg_metrics, 1)
        if dataset_name in args.ood_datasets:
            overall_ood_average_meter.aggregate(avg_metrics, 1)
            
        for category, metrics_obj in avg_metrics.items():  # category is 'whole' or 'high'
            metrics = metrics_obj.values  # Directly access the values attribute of Metrics object
            for metric_name, value in metrics.items():
                if isinstance(value, list):  # Check if the metric value is a list
                    for i, val in enumerate(value):
                        # Append an index or threshold identifier to the metric name
                        writer.add_scalar(f'Eval/{metric_name}_{i}_{category}_{dataset_name}', val, i_iter)
                else:
                    writer.add_scalar(f'Eval/{metric_name}_{category}_{dataset_name}', value, i_iter)
                    
    # Handling overall metrics
    # This returns a dictionary of Metrics objects
    overall_avg_metrics = overall_average_meter.average()  # Aggregate metrics from each dataset
    print_metrics(overall_avg_metrics, logger, "OVERALL")

    for category, metrics_obj in overall_avg_metrics.items():  # category is 'whole' or 'high'
        metrics = metrics_obj.values  # Directly access the values attribute of Metrics object
        for metric_name, value in metrics.items():
            if isinstance(value, list):  # Check if the metric value is a list
                for i, val in enumerate(value):
                    # Append an index or threshold identifier to the metric name
                    writer.add_scalar(f'OVERALL_Eval/{metric_name}_{i}_{category}', val, i_iter)
            else:
                writer.add_scalar(f'OVERALL_Eval/{metric_name}_{category}', value, i_iter)
                
    overall_ood_avg_metrics = overall_ood_average_meter.average()  # Aggregate metrics from each dataset
    ood_avg_mae = overall_ood_avg_metrics['whole'].values['mae']
    print_metrics(overall_ood_avg_metrics, logger, "OOD_OVERALL")
    results = {'HE': ood_avg_mae}

    if number_ss_data!=0:
        average_iou = total_iou/number_ss_data
        results['SS'] = average_iou
        logger.info(f'[Overall miou]: {average_iou}')
    for category, metrics_obj in overall_ood_avg_metrics.items():  # category is 'whole' or 'high'
        metrics = metrics_obj.values  # Directly access the values attribute of Metrics object
        for metric_name, value in metrics.items():
            if isinstance(value, list):  # Check if the metric value is a list
                for i, val in enumerate(value):
                    # Append an index or threshold identifier to the metric name
                    writer.add_scalar(f'OOD_OVERALL_Eval/{metric_name}_{i}_{category}', val, i_iter)
            else:
                writer.add_scalar(f'OOD_OVERALL_Eval/{metric_name}_{category}', value, i_iter)
    return results

if __name__ == '__main__':
    main()