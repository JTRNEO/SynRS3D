import numpy as np
import torch
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

from models.dpt import DPT_DINOv2
import os
import argparse
from tqdm import tqdm
from osgeo import gdal
from utils import geo_func

def save_tiff_with_geo_info(input_path, output_path, data):
    """
    Saves the data as a TIFF file with the geographic information from the input TIFF.
    Args:
        input_path: Path to the original TIFF file.
        output_path: Path where the output TIFF file will be saved.
        data: The image data to save. Can be 2D (single channel) or 3D (multi-channel).
    """
    input_ds = gdal.Open(input_path, gdal.GA_ReadOnly)
    geo_transform = input_ds.GetGeoTransform()
    projection = input_ds.GetProjection()

    driver = gdal.GetDriverByName('GTiff')

    if len(data.shape) == 2:  # Single channel image
        height, width = data.shape
        channels = 1
        data_type = gdal.GDT_Byte
    else:  # Multi-channel image
        height, width, channels = data.shape
        data_type = gdal.GDT_Byte  # Use Byte for multi-channel images
        # Scale the data to 0-255 range for uint8
        data = (data * 255).astype(np.uint8)

    output_ds = driver.Create(output_path, width, height, channels, data_type)
    output_ds.SetGeoTransform(geo_transform)
    output_ds.SetProjection(projection)

    if channels == 1:
        output_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(channels):
            output_ds.GetRasterBand(i + 1).WriteArray(data[:, :, i])

    output_ds.FlushCache()
    output_ds = None

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inference of Semantic Segmentation")
    parser.add_argument("--data_dir", type=str, default='path/to/your/input.tif',
                        help="Path of the image.")
    parser.add_argument("--output_path", type=str, default='path/to/your/ouput.tif',
                        help="Path to the output director.")
    parser.add_argument("--restore_from", type=str, default='./pretrain/RS3DAda_vitl_DPT_segmentation.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--patch_size", type=int, default=1022,
                        help="Make sure it is divisible by 14")
    parser.add_argument("--overlap", type=int, default=511,
                        help="overlap size.")
    parser.add_argument("--use_tta", action='store_true', help="Whether to use test-time augmentation.")
    return parser.parse_args()

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    head_configs = [
        {'name': 'regression', 'nclass': 1},
        {'name': 'segmentation', 'nclass': 8}
    ]
    model = DPT_DINOv2(encoder='vitl', head_configs=head_configs, pretrained=False)
    model.load_state_dict(torch.load(args.restore_from))
    model.eval()
    model.cuda()

    # Load and process the image in patches
    # process large Geotiff image
    img0 = geo_func.load_multiband(args.data_dir)

    if img0.shape[2] > 3:
        img0 = img0[:,:,[0,1,2]]
    height = img0.shape[0]
    width = img0.shape[1]

    patch_size = args.patch_size
    stride = args.overlap
    C = int(np.ceil( (width - patch_size) / stride ) + 1)
    R = int(np.ceil( (height - patch_size) / stride ) + 1)
    
    # weight matrix B for avoiding boundaries of patches
    w = patch_size
    if patch_size > stride:
        s1 = stride
        s2 = w - s1
        d = 1/(1+s2)
        B1 = np.ones((w,w))
        B1[:,s1::] = np.dot(np.ones((w,1)),(-np.arange(1,s2+1)*d+1).reshape(1,s2))
        B2 = np.flip(B1)
        B3 = B1.T
        B4 = np.flip(B3)
        B = B1*B2*B3*B4
    else:
        B = np.ones((w,w))

    img1 = np.zeros((patch_size+stride*(R-1), patch_size+stride*(C-1),3))
    img1[0:height,0:width,:] = img0.copy()

    pred_all = np.zeros((8,patch_size+stride*(R-1), patch_size+stride*(C-1)))
    weight = np.zeros((patch_size+stride*(R-1), patch_size+stride*(C-1)))
    patch_transformed = Compose([
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
        ToTensorV2()
    ])
    for r in tqdm(range(R), desc="Row"):
        for c in tqdm(range(C), desc="Column", leave=False):
            img = img1[r*stride:r*stride+patch_size,c*stride:c*stride+patch_size,:].copy().astype(np.float32)
            if args.use_tta:
                imgs = [
                    img.copy(),
                    img[:, ::-1, :].copy(),
                    img[::-1, :, :].copy(),
                    img[::-1, ::-1, :].copy()
                ]
                input = torch.cat([patch_transformed(image=x)['image'].cuda().unsqueeze(0) for x in imgs], dim=0).float().cuda()
            else:
                input = patch_transformed(image=img)['image'].cuda().unsqueeze(0).float().cuda()

            pred = []
            with torch.no_grad():
                outputs = model(input)
                if args.use_tta:
                    output = outputs.get('segmentation', None).data.cpu().numpy()
                    pred = (output[0, :, :, :] + output[1, :, :, ::-1] + output[2, :, ::-1, :] + output[3, :, ::-1, ::-1])/4
                else:
                    pred = outputs.get('segmentation', None).data.cpu().numpy().squeeze(0)

            pred_all[:,r*stride:r*stride+patch_size,c*stride:c*stride+patch_size] += pred.copy()*B
            weight[r*stride:r*stride+patch_size,c*stride:c*stride+patch_size] += B
            
    for b in range(8):
        pred_all[b,:,:] = pred_all[b,:,:]/weight

    pred_all = np.argmax(pred_all, axis=0).astype(np.uint8)
    pred_all = pred_all[0:height, 0:width]
             
    save_tiff_with_geo_info(args.data_dir, args.output_path, pred_all)

if __name__ == '__main__':
    main()
