from PIL import Image
import numpy as np
import torch

#######ClassMix Mask
def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(label)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks

def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask
  
#########CutMix Mask

def generate_cutmix_masks(B, H, W, alpha=0.25):
    """
    Generate a list of CutMix masks for the specified batch size and image dimensions.
    
    Parameters:
    B (int): Batch size, the number of masks to generate.
    H (int): Height of the images.
    W (int): Width of the images.
    alpha (float): Parameter for the beta distribution.

    Returns:
    List[torch.Tensor]: A list of binary mask tensors, each with the height and width specified.
    """
    masks = []
    for _ in range(B):
        mask = torch.ones(1, 1, H, W)

        # Sample lambda and calculate patch dimensions
        lambda_ = np.random.beta(alpha, alpha)
        rx = np.random.uniform(0, W)
        ry = np.random.uniform(0, H)
        rw = W * np.sqrt(1 - lambda_)
        rh = H * np.sqrt(1 - lambda_)
        x1 = int(max(rx - rw / 2, 0))
        x2 = int(min(rx + rw / 2, W))
        y1 = int(max(ry - rh / 2, 0))
        y2 = int(min(ry + rh / 2, H))

        mask[:, :, y1:y2, x1:x2] = 0
        masks.append(mask)
    return masks
  
def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target