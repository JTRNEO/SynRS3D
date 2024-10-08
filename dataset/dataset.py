import numpy as np
import os
from torch.utils import data
from PIL import Image
from typing import Optional, List, Dict
import random
import albumentations as A
from utils.datasets_config import get_dataset_category,labelmap,combination_relabel_rules, normal_relabel_rules

class MultiTaskDataSet(data.Dataset):
    def __init__(self, root, 
                 is_training=True, 
                 images_file: List[str]=['train.txt', 'test_syn.txt', 'test.txt'], 
                 transforms=None, 
                 max_iters=None, 
                 max_da_images=500000,
                 multi_task=False, 
                 combine_class=True,
                 apply_da: List[str]=[],
                 da_aug_paras: Optional[Dict] = None,
                 tgt_root_dir: List[str]=[],
                 ignore_label=255):
        
        self.root = root
        self.is_training = is_training
        self.multi_task = multi_task
        self.combine_class = combine_class
        
        self.apply_da = apply_da if self.is_training else []
        self.da_aug_paras = da_aug_paras
        self.train_file = images_file[0]
        self.test_file = images_file[1]
        self.tgt_file = images_file[2]
        
        self.tgt_files_path = []
        self.max_da_images = max_da_images
        
        if self.apply_da:
            for dir in tgt_root_dir:
                with open(os.path.join(dir, self.tgt_file), 'r') as file:
                    tgt_img_ids_in_dir = [line.strip() for line in file]
                    # Shuffle the list first
                    random.shuffle(tgt_img_ids_in_dir)

                    # If more images than max_images_per_dir, truncate the list
                    if len(tgt_img_ids_in_dir) > self.max_da_images:
                        tgt_img_ids_in_dir = tgt_img_ids_in_dir[:self.max_da_images]
                        
                    self.tgt_files_path.extend(os.path.join(dir, f"opt/{name}.tif") for name in tgt_img_ids_in_dir)
                    
        self.list_path = [os.path.join(data, self.train_file if self.is_training else self.test_file) for data in self.root]
        
        self.datasetname = set([os.path.basename(i) for i in self.root])
        self.transforms = transforms

        self.ignore_label = ignore_label
        self.img_ids = []
        self.files = []

        for root_dir in self.root:
            list_path = os.path.join(root_dir, self.train_file if self.is_training else self.test_file)
            with open(list_path, 'r') as file:
                img_ids_in_dir = [line.strip() for line in file]
                self.img_ids.extend(img_ids_in_dir)
                if self.multi_task:
                    self.files.extend([{'img': os.path.join(root_dir, f"opt/{name}.tif"),
                                        'dsm': os.path.join(root_dir, f"gt_nDSM/{name}.tif"),
                                        'ss_mask': os.path.join(root_dir, f"gt_ss_mask/{name}.tif"),
                                        'name': name} for name in img_ids_in_dir])
                else:
                    self.files.extend([{'img': os.path.join(root_dir, f"opt/{name}.tif"),
                                    'dsm': os.path.join(root_dir, f"gt_nDSM/{name}.tif"),
                                    'name': name} for name in img_ids_in_dir])
        random.shuffle(self.files)
        if max_iters is not None:
            self.files = (self.files * int(np.ceil(float(max_iters) / len(self.files))))[:max_iters]
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        try:
            # Load image and convert to float32 RGB
            src_image = np.array(Image.open(datafiles["img"]).convert('RGB'))
            if self.apply_da and self.tgt_files_path:
                tgt_img_path = random.choice(self.tgt_files_path)
                tgt_img = np.array(Image.open(tgt_img_path).convert('RGB'))
                # Define augmentation methods with their respective configurations
                da_aug_template = {
                    'FDA': A.FDA(reference_images=[tgt_img], beta_limit=self.da_aug_paras['FDA']['beta_limit'], p=1, read_fn=lambda x: x), 
                    'HM': A.HistogramMatching(reference_images=[tgt_img], blend_ratio=self.da_aug_paras['HM']['blend_ratio'], read_fn=lambda x: x, p=1), 
                    'PDA': A.PixelDistributionAdaptation(reference_images=[tgt_img], blend_ratio=self.da_aug_paras['PDA']['blend_ratio'], read_fn=lambda x: x, transform_type=self.da_aug_paras['PDA']['transform_type'], p=1)
                }
                # Filter the augmentations to only include those specified in apply_da
                selected_augs = [da_aug_template[da] for da in self.apply_da if da in da_aug_template]
                
                if selected_augs:
                    # Apply one of the selected augmentations randomly
                    da_augmentation = A.Compose([A.OneOf(selected_augs, p=0.8)])
                    src_image = da_augmentation(image=src_image)['image']
                else:
                    raise ValueError('no such a da augmentation')
                
            image = np.array(src_image, dtype=np.float32)
            # Load and preprocess DSM
            dsm = np.array(Image.open(datafiles["dsm"]), dtype=np.float32)
            dsm[np.isnan(dsm) | (dsm > 450)] = 0
            dsm = np.expand_dims(dsm, axis=-1)

            # Initialize the result dict with common entries
            result_dict = {"image": image, "size": np.array(image.shape[:2]), "name": datafiles["name"], "dsm": dsm}

            # Handling semantic segmentation mask if needed
            if self.multi_task:
                dataset_category = get_dataset_category(self.datasetname)
                if dataset_category is not None:
                    relabel_rules = combination_relabel_rules if self.combine_class else normal_relabel_rules
                    ss_mask = np.array(Image.open(datafiles["ss_mask"]), dtype=np.uint8)
                    ss_mask = labelmap(ss_mask, relabel_rules[dataset_category])
                    ss_mask = np.array(np.expand_dims(ss_mask, axis=-1), dtype=np.float32)
                    result_dict["ss_mask"] = ss_mask
                else:
                    raise ValueError('No such a ss dataset type!')

            # Apply transformations if specified
            if self.transforms:
                masks_to_transform = [result_dict.get("dsm"), result_dict.get("ss_mask", None)]
                augmented = self.transforms(image=image, masks=[mask for mask in masks_to_transform if mask is not None])
                image = augmented['image']
                transformed_masks = augmented['masks']
                
                # Update the result dict with transformed data
                result_dict["image"] = image
                if "dsm" in result_dict:
                    result_dict["dsm"] = transformed_masks.pop(0)
                if "ss_mask" in result_dict:
                    result_dict["ss_mask"] = transformed_masks.pop(0)

                # Ensure correct dimension ordering (channel-first for PyTorch)
                for key in ["dsm", "ss_mask"]:
                    if key in result_dict:
                        result_dict[key] = result_dict[key].permute(2, 0, 1)

            return result_dict
        
        except IOError as e:
            print(f"Error reading file {datafiles['img']} or {datafiles['dsm']}: {e}")
            return None

class PesudoDataSet(data.Dataset):
    def __init__(self, root, 
                 images_file='train.txt', 
                 max_da_images = 500000,
                 transforms=None, 
                 max_iters=None):
        self.root = root
        self.train_file = images_file
        self.list_path = [os.path.join(data, self.train_file) for data in self.root]

        self.datasetname = set([os.path.basename(i) for i in self.root])
        self.transforms = transforms
        self.img_ids = []
        self.files = []
        self.max_da_images = max_da_images
        for root_dir in self.root:
            list_path = os.path.join(root_dir, self.train_file)
            with open(list_path, 'r') as file:
                img_ids_in_dir = [line.strip() for line in file]
                # Shuffle the list first
                random.shuffle(img_ids_in_dir)

                # If more images than max_images_per_dir, truncate the list
                if len(img_ids_in_dir) > self.max_da_images:
                    img_ids_in_dir = img_ids_in_dir[:self.max_da_images]
                self.img_ids.extend(img_ids_in_dir)
                self.files.extend([{'img': os.path.join(root_dir, f"opt/{name}.tif"),
                                    'name': name} for name in img_ids_in_dir])
        random.shuffle(self.files)
        if max_iters is not None:
            self.files = (self.files * int(np.ceil(float(max_iters) / len(self.files))))[:max_iters]
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        try:
            # Load image and convert to float32 RGB
            src_image = np.array(Image.open(datafiles["img"]).convert('RGB'))
                
            image = np.array(src_image, dtype=np.float32)

            # Initialize the result dict with common entries
            result_dict = {"image": image, "size": np.array(image.shape[:2]), "name": datafiles["name"]}

            # Apply transformations if specified
            if self.transforms:
                augmented = self.transforms(image=image)
                image = augmented['image']
                
                # Update the result dict with transformed data
                result_dict["image"] = image

            return result_dict
        
        except IOError as e:
            print(f"Error reading file {datafiles['img']} or {datafiles['dsm']}: {e}")
            return None
        
class OEMDataSet(data.Dataset):
    def __init__(self, root, 
                 is_training=True, 
                 images_file: List[str]=['train.txt', 'test_syn.txt', 'test.txt'], 
                 transforms=None, 
                 max_iters=None, 
                 combine_class=True,
                 apply_da: List[str]=[],
                 da_aug_paras: Optional[Dict] = None,
                 tgt_root_dir: List[str]=[],
                 ignore_label=255
                 ):
        self.root = root
        self.is_training = is_training
        self.combine_class = combine_class
        
        self.apply_da = apply_da if self.is_training else []
        self.da_aug_paras = da_aug_paras
        self.train_file = images_file[0]
        self.test_file = images_file[1]
        self.tgt_file = images_file[2]
        self.tgt_files_path = []
        if self.apply_da:
            for dir in tgt_root_dir:
                with open(os.path.join(dir, self.tgt_file), 'r') as file:
                    tgt_img_ids_in_dir = [line.strip() for line in file]
                    self.tgt_files_path.extend(os.path.join(dir, f"opt/{name}.tif") for name in tgt_img_ids_in_dir)

        self.list_path = [os.path.join(data, self.train_file if self.is_training else self.test_file) for data in self.root]
        
        self.datasetname = set([os.path.basename(i) for i in self.root])
        self.transforms = transforms

        self.ignore_label = ignore_label
        self.img_ids = []
        self.files = []
        
        for root_dir in self.root:
            list_path = os.path.join(root_dir, self.train_file if self.is_training else self.test_file)
            with open(list_path, 'r') as file:
                img_ids_in_dir = [line.strip() for line in file]
                self.img_ids.extend(img_ids_in_dir)
                self.files.extend([{'img': os.path.join(root_dir, f"opt/{name}.tif"),
                                    'ss_mask': os.path.join(root_dir, f"gt_ss_mask/{name}.tif"),
                                    'name': name} for name in img_ids_in_dir])
        random.shuffle(self.files)
        
        if max_iters is not None:
            self.files = (self.files * int(np.ceil(float(max_iters) / len(self.files))))[:max_iters]
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        try:
            # Load image and convert to float32 RGB
            src_image = np.array(Image.open(datafiles["img"]).convert('RGB'))
            if self.apply_da and self.tgt_files_path:
                tgt_img_path = random.choice(self.tgt_files_path)
                tgt_img = np.array(Image.open(tgt_img_path).convert('RGB'))
                # Define augmentation methods with their respective configurations
                da_aug_template = {
                    'FDA': A.FDA(reference_images=[tgt_img], beta_limit=self.da_aug_paras['FDA']['beta_limit'], p=1, read_fn=lambda x: x), 
                    'HM': A.HistogramMatching(reference_images=[tgt_img], blend_ratio=self.da_aug_paras['HM']['blend_ratio'], read_fn=lambda x: x, p=1), 
                    'PDA': A.PixelDistributionAdaptation(reference_images=[tgt_img], blend_ratio=self.da_aug_paras['PDA']['blend_ratio'], read_fn=lambda x: x, transform_type=self.da_aug_paras['PDA']['transform_type'], p=1)
                }
                # Filter the augmentations to only include those specified in apply_da
                selected_augs = [da_aug_template[da] for da in self.apply_da if da in da_aug_template]
                
                if selected_augs:
                    # Apply one of the selected augmentations randomly
                    da_augmentation = A.Compose([A.OneOf(selected_augs, p=0.8)])
                    src_image = da_augmentation(image=src_image)['image']
                else:
                    raise ValueError('no such a da augmentation')
                
            image = np.array(src_image, dtype=np.float32)

            # Initialize the result dict with common entries
            result_dict = {"image": image, "size": np.array(image.shape[:2]), "name": datafiles["name"]}

            dataset_category = get_dataset_category(self.datasetname)
            if dataset_category is not None:
                relabel_rules = combination_relabel_rules if self.combine_class else normal_relabel_rules
                ss_mask = np.array(Image.open(datafiles["ss_mask"]), dtype=np.uint8)
                ss_mask = labelmap(ss_mask, relabel_rules[dataset_category])
                ss_mask = np.array(np.expand_dims(ss_mask, axis=-1), dtype=np.float32)
                result_dict["ss_mask"] = ss_mask

            # Apply transformations if specified
            if self.transforms:
                masks_to_transform = [result_dict.get("ss_mask", None)]
                augmented = self.transforms(image=image, masks=[mask for mask in masks_to_transform if mask is not None])
                image = augmented['image']
                transformed_masks = augmented['masks']
                
                # Update the result dict with transformed data
                result_dict["image"] = image
                if "ss_mask" in result_dict:
                    result_dict["ss_mask"] = transformed_masks.pop(0)
                # Ensure correct dimension ordering (channel-first for PyTorch)
                for key in ["ss_mask"]:
                    if key in result_dict:
                        result_dict[key] = result_dict[key].permute(2, 0, 1)

            return result_dict
        
        except IOError as e:
            print(f"Error reading file {datafiles['img']} or {datafiles['dsm']}: {e}")
            return None