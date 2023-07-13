import pandas as pd
import numpy as np
from iou import intersection_over_union
import torch
from matplotlib import pyplot as plt
import numpy as np

def mean_average_precision(data: pd.DataFrame, iou_threshold=0.4, plot_curves=False):
    '''Calculates mAP for each type of object based on dataframe'''
    epsilon = 1e-9
    class_list = data['obj_type'].unique()
    APs = {}
    
    #preparing axes for plots
    nrows = int(np.ceil(len(class_list) / 3))
    ncols = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 10))
    for i in range(nrows):
        for j in range(ncols):
            axs[i, j].set_visible(False)
    ax_x = 0
    ax_y = 0
    
    for type in class_list:
        cur_class_df = data[data['obj_type'] == type]
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
        
        APs[type] = float(torch.trapz(precisions, recalls))
        
        #plotting current class AP curve
        axs[ax_x, ax_y].set_visible(True)
        axs[ax_x, ax_y].plot(recalls, precisions, label=f'AP = {round(APs[type], 4)}')
        axs[ax_x, ax_y].set_title(type)
        axs[ax_x, ax_y].legend(loc='upper right')
        axs[ax_x, ax_y].set_xlabel('recall')
        axs[ax_x, ax_y].set_ylabel('precision')
        ax_y += 1
        if ax_y == 3:
            ax_y = 0
            ax_x += 1

    return APs