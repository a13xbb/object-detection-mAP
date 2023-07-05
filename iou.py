def intersection_over_union(cords_pred: list, cords_gt: list) -> float:
    '''cords argument looks like this: [x, y, width, height]'''
    boxA = [cords_pred[0], cords_pred[1], cords_pred[0] + cords_pred[2], cords_pred[1] + cords_pred[3]]
    boxB = [cords_gt[0], cords_gt[1], cords_gt[0] + cords_gt[2], cords_gt[1] + cords_gt[3]]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inner_area = max(0, xB - xA) * max(0, yB - yA)
    
    if inner_area == 0:
        return 0
    
    boxA_area = cords_pred[2] * cords_pred[3]
    boxB_area = cords_gt[2] * cords_gt[3]
    
    iou = inner_area / float(boxA_area + boxB_area - inner_area)
    
    return iou