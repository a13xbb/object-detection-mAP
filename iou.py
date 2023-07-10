def intersection_over_union(cords_pred: list, cords_gt: list) -> float:
    '''cords argument looks like this: [x, y, width, height]'''
    box_a = [cords_pred[0], cords_pred[1], cords_pred[0] + cords_pred[2], cords_pred[1] + cords_pred[3]]
    box_b = [cords_gt[0], cords_gt[1], cords_gt[0] + cords_gt[2], cords_gt[1] + cords_gt[3]]
    
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    
    inner_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    
    if inner_area == 0:
        return 0
    
    box_a_area = cords_pred[2] * cords_pred[3]
    box_b_area = cords_gt[2] * cords_gt[3]
    
    iou = inner_area / float(box_a_area + box_b_area - inner_area)
    
    return iou