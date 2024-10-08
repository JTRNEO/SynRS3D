import torch
import math
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x."""
    return np.log(x) / math.log(10)

def calculate_f1_score_for_heights(output, target, threshold, accuracy_threshold=1.25):
    significant_actual = target > threshold
    significant_pred = output > threshold
    not_significant_actual = ~significant_actual
    not_significant_pred = ~significant_pred

    maxRatio = np.maximum(output / target, target / output)
    tolerance = np.abs(output-target) < threshold/2
    
    TP = ((significant_pred & significant_actual) & (maxRatio < accuracy_threshold)).sum()
    FP = (significant_pred & not_significant_actual).sum()
    FN = (not_significant_pred & significant_actual).sum()
    TN = ((not_significant_pred & not_significant_actual) & (tolerance)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    # Calculate the components for Cohen's Kappa
    total = TP + FP + FN + TN
    Po = accuracy  # Observed agreement
    Pe = (((TP + FP) / total) * ((TP + FN) / total)) + (((FN + TN) / total) * ((FP + TN) / total))

    # Calculate Cohen's Kappa
    kappa = (Po - Pe) / (1 - Pe) if (1 - Pe) != 0 else 0

    return accuracy, precision, recall, f1_score, kappa
        
class Metrics:
    def __init__(self, calculate_f1=False):
        self.calculate_f1 = calculate_f1
        self.reset()

    def reset(self):
        self.values = {
            "mae": 0, "rmse": 0, "mse": 0, "log": 0,
            "delta": [0, 0, 0]
        }
        if self.calculate_f1:
            self.values.update({
                "OA": [0, 0, 0], "precision": [0, 0, 0], "recall": [0, 0, 0], "f1_score": [0, 0, 0], "Kappa": [0, 0, 0], "ac": 0
            })
        self.valid = True
        
    def update_ordinal_ac(self, preds, masks):
        # Assuming preds and targets are already flattened and processed if necessary
        self.values["ac"] = accuracy_score(preds.view(), masks)
        
    def calculate_metrics(self, output, target, threshold=0):
        if np.any(target > threshold) and np.any(output > threshold):
            valid_mask = ((target > threshold) + (output > threshold)) > 0
            output_valid = output[valid_mask]
            target_valid = target[valid_mask]
            # Ensure images are at least 7x7
            # if output_valid.size < 49 or target_valid.size < 49 and np.all(target <= 1):
            if output_valid.size < 49 or target_valid.size < 49:
                print("Images are too small for SSIM calculation. Ensure images are at least 7x7.")
                self.valid = False
                return
            abs_diff = np.abs(output_valid - target_valid)
            log_diff = np.abs(log10(output_valid) - log10(target_valid))
            
            self.values["mae"] = np.mean(abs_diff)
            self.values["mse"] = np.mean(abs_diff ** 2)
            self.values["rmse"] = np.sqrt((abs_diff ** 2).mean())
            self.values["log"] = np.mean(log_diff)
            
            maxRatio = np.maximum(output_valid / target_valid, target_valid / output_valid)
            self.values["delta"] = [
                (maxRatio < 1.25).mean(),
                (maxRatio < 1.25**2).mean(),
                (maxRatio < 1.25**3).mean()
            ]
            
            if self.calculate_f1:
                for i, acc_thresh in enumerate([1.25, 1.25**2, 1.25**3]):
                    acc, precision, recall, f1_score, kappa = calculate_f1_score_for_heights(output_valid, target_valid, 1, acc_thresh)
                    self.values["OA"][i] = acc
                    self.values["precision"][i] = precision
                    self.values["recall"][i] = recall
                    self.values["f1_score"][i] = f1_score
                    self.values["Kappa"][i] = kappa
        else:
            self.valid = False

class Result:
    def __init__(self):
        self.metrics = {"whole": Metrics(calculate_f1=True), "low": Metrics(), "mid": Metrics(), "high": Metrics()}
    
    def update(self, output, target):
        output[output <= 0] = 1e-6
        target[target <= 0] = 1e-6
        self.metrics["whole"].calculate_metrics(output, target, threshold=0)
        self.metrics["low"].calculate_metrics(output, target, threshold=1)
        self.metrics["mid"].calculate_metrics(output, target, threshold=2)
        self.metrics["high"].calculate_metrics(output, target, threshold=3)
        
    def update_ordinal_ac(self, pred_masks, ndsm_masks):
        # Assuming pred_masks and ndsm_masks are processed if necessary before this call
        self.metrics["whole"].update_ordinal_ac(pred_masks, ndsm_masks)
        
    def get_metrics(self):
        return {k: v.values for k, v in self.metrics.items()}

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.whole = Metrics(calculate_f1=True)  # Calculate F1 for whole
        self.low = Metrics()
        self.mid = Metrics()
        self.high = Metrics()  # No F1 for high
        self.total = {"whole": 0, "low": 0, "mid": 0, "high": 0}  # To keep track of total samples for weighted average

    def update(self, result, n=1):
        for key in ["whole", "low", "mid", "high"]:
            if result.metrics[key].valid:
                self.total[key] += n
                for metric, value in result.metrics[key].values.items():
                    if metric in ["OA", "precision", "recall", "f1_score", "Kappa"] and key != "whole": continue  # Skip F1 metrics for 'high'
                    if isinstance(value, list):  # Handle lists (precision, recall, f1_score)
                        for i in range(3):
                            self.__dict__[key].values[metric][i] += result.metrics[key].values[metric][i] * n
                    else:
                        self.__dict__[key].values[metric] += value * n
                        
    def aggregate(self, avg_metrics, n=1):
        for key, metrics_obj in avg_metrics.items():
            for metric_name, value in metrics_obj.values.items():
                if isinstance(value, list):
                    if metric_name not in self.__dict__[key].values:
                        self.__dict__[key].values[metric_name] = [0] * len(value)
                    for i in range(len(value)):
                        self.__dict__[key].values[metric_name][i] += value[i] * n
                else:
                    self.__dict__[key].values[metric_name] += value * n
            self.total[key] += n
            
    def average(self):
        avg = {"whole": Metrics(calculate_f1=True), "low": Metrics(), "mid": Metrics(), "high": Metrics()}
        for key in ["whole", "low", "mid", "high"]:
            for metric, value in self.__dict__[key].values.items():
                if isinstance(value, list):  # Handle lists for precision, recall, f1_score
                    avg[key].values[metric] = [v / self.total[key] if self.total[key] > 0 else 0 for v in value]
                else:
                    avg[key].values[metric] = value / self.total[key] if self.total[key] > 0 else 0
        return avg
