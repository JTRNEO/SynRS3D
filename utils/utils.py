import torch
from albumentations import ColorJitter, GaussianBlur, Compose, Normalize, OneOf
from albumentations.pytorch import ToTensorV2

def get_transforms():
    # Define transformations that can apply ColorJitter, GaussianBlur, both, or none (NoOp)
    transform = Compose([
        OneOf([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1),
            GaussianBlur(blur_limit=(3, 7), p=1),
            Compose([
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                GaussianBlur(blur_limit=(3, 7))
            ])
        ]),
        Normalize(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), max_pixel_value=1, always_apply=True),
        ToTensorV2()
    ])
    return transform

def denormalize(tensor):
    # Mean and std must be reshaped to [1, C, 1, 1] to match the tensor dimensions [B, C, H, W]
    mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    # Reverse the normalization
    denormalized_tensor = (tensor * std) + mean

    # Ensure the output is clipped to the range [0, 255] since the original images were likely in this range
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 255)

    return denormalized_tensor.long()

def adjust_learning_rate(optimizer, base_lr, i_iter, num_steps, power, warmup_steps, warmup_mode='linear', decay_mode='poly'):
    """Adjusts the learning rate with configurable warm-up and decay phases."""
    for param_group in optimizer.param_groups:
        base_lr = param_group['init_lr'] 
        if i_iter < warmup_steps:
            # Warm-up phase
            if warmup_mode == 'linear':
                lr = base_lr * (i_iter / warmup_steps)
            elif warmup_mode == 'poly':
                lr = base_lr * ((i_iter / warmup_steps) ** power)
        else:
            # Decay phase
            if decay_mode == 'linear':
                lr = base_lr * (1 - (i_iter - warmup_steps) / (num_steps - warmup_steps))
            elif decay_mode == 'poly':
                lr = base_lr * ((1 - (i_iter - warmup_steps) / (num_steps - warmup_steps)) ** power)

        param_group['lr'] = lr
    return [pg['lr'] for pg in optimizer.param_groups]

def update_ema(teacher, student, iter, alpha):
    alpha_teacher = min(1 - 1 / (iter + 1), alpha)
    for ema_param, param in zip(teacher().parameters(),
                                student().parameters()):
        if not param.data.shape:  # scalar tensor
            ema_param.data = \
                alpha_teacher * ema_param.data + \
                (1 - alpha_teacher) * param.data
        else:
            ema_param.data[:] = \
                alpha_teacher * ema_param[:].data[:] + \
                (1 - alpha_teacher) * param[:].data[:]
                
def print_metrics(average_metrics, logger, dataset_name):
    logger.info(f'[Eval]: {dataset_name} nDSM Estimate metrics:')

    # Determine the maximum width for formatting based on metric names and values
    max_metric_name_len = max(len(metric) for category in average_metrics.values() for metric in category.values.keys())
    max_value_len = max(len(", ".join(f"{v:.3f}" for v in value)) if isinstance(value, list) else len(f"{value:.3f}") 
                        for category in average_metrics.values() for value in category.values.values())

    # Header
    categories = list(average_metrics.keys())  # e.g., ['whole', 'high']
    header_parts = ["Metric".ljust(max_metric_name_len)] + [cat.ljust(max_value_len) for cat in categories]
    header = " | ".join(header_parts)
    logger.info(header)
    logger.info("-" * len(header))

    # Iterate through metrics based on the order in the 'whole' category
    reference_category = 'whole'
    for metric_name in average_metrics[reference_category].values.keys():
        row_parts = [metric_name.ljust(max_metric_name_len)]
        for category in categories:
            value = average_metrics[category].values.get(metric_name)
            if isinstance(value, list):
                formatted_value = ", ".join(f"{v:.3f}" for v in value).ljust(max_value_len)
            elif value is not None:
                formatted_value = f"{value:.3f}".ljust(max_value_len)
            else:
                formatted_value = "N/A".ljust(max_value_len)
            row_parts.append(formatted_value)
        row = " | ".join(row_parts)
        logger.info(row)

    logger.info("-" * len(header))