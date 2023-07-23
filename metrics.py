import pandas as pd
import numpy as np
from iou import intersection_over_union
import torch
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

def get_metrics (data: pd.DataFrame, obj_type: str, iou_threshold=0.4) -> tuple:
    '''Calculates mAP for each type of object based on dataframe
       Returns AP, number of true objects, number of TP samples, number of FP samples,
       precisions, recalls'''
    epsilon = 1e-9
    class_list = data['obj_type'].unique()
    if obj_type not in class_list:
        raise Exception('No such obj_type in passed dataframe')
    
    cur_class_df = data[data['obj_type'] == obj_type]
    cur_class_df = cur_class_df.sort_values('confidence', ascending=False)
    num_detections = len(cur_class_df.loc[(cur_class_df['w_pred'] != 0) & (cur_class_df['h_pred'] != 0)])
    num_true_objects = len(cur_class_df.loc[(cur_class_df['w_gt'] != 0) & (cur_class_df['h_gt'] != 0)])
    
    TP = torch.zeros(num_detections)
    FP = torch.zeros(num_detections)
    
    detection_idx = 0
    for index, row in cur_class_df.iterrows():     
        if row['w_pred'] != 0 and row['h_pred'] != 0:
            cords_pred = [row['x_pred'], row['y_pred'], row['w_pred'], row['h_pred']]
            cords_gt = [row['x_gt'], row['y_gt'], row['w_gt'], row['h_gt']]
            iou = intersection_over_union(cords_pred, cords_gt)
            if iou >= iou_threshold:
                TP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
            
            detection_idx += 1
            
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    
    recalls = TP_cumsum / (num_true_objects + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    recalls = torch.cat((torch.tensor([0]), recalls))
    precisions = torch.cat((torch.tensor([1]), precisions))
    
    AP = float(torch.trapz(precisions, recalls))
    TP_cnt = TP_cumsum[-1]
    FP_cnt = FP_cumsum[-1]

    return AP, num_true_objects, num_detections, int(TP_cnt), int(FP_cnt), precisions, recalls


def plot_pr_curve(precisions, recalls, title) -> None:
    AP = float(torch.trapz(precisions, recalls))
    plt.plot(recalls, precisions, label=f'AP = {round(AP, 4)}')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    return


def get_gt_and_proba(data: pd.DataFrame, obj_type: str, iou_threshold=0.4) -> tuple:
    epsilon = 1e-9
    class_list = data['obj_type'].unique()
    if obj_type not in class_list:
        raise Exception('No such obj_type in passed dataframe')
    
    cur_class_df = data[data['obj_type'] == obj_type]
    
    y_true = []
    y_proba = []
    
    for index, row in cur_class_df.iterrows():     
        if row['w_pred'] != 0 and row['h_pred'] != 0:
            cords_pred = [row['x_pred'], row['y_pred'], row['w_pred'], row['h_pred']]
            cords_gt = [row['x_gt'], row['y_gt'], row['w_gt'], row['h_gt']]
            iou = intersection_over_union(cords_pred, cords_gt)
            if iou >= iou_threshold:
                y_proba.append(row['confidence'])
            else:
                y_proba.append(0.0)
        else:
            y_proba.append(0.0)
            
        if row['w_gt'] != 0 and row['h_gt'] != 0:
            y_true.append(1)
        else:
            y_true.append(0)

    return y_true, y_proba


def plot_roc_curve(y_true: list, y_proba: list):
    if len(np.unique(y_true)) == 1:
        y_true[0] = int(not(y_true[0]))
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    plt.plot(fpr, tpr, label=f'AUC: {roc_auc_score(y_true, y_proba)}')
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), 'r--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.show()
    return