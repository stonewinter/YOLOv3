import numpy as xp

def CalcIoU(rects1, rects2):
    """
    To calculate IoU between multiple rects and multiple rects
    :param rects1: format(n, 4) = [[x1, y1, x2, y2],
                                   [x1, y1, x2, y2],
                                   [...]]
    :param rects2: format(m, 4) = [[x1, y1, x2, y2],
                                   [x1, y1, x2, y2],
                                   [...]]
    :return: format(n, m) = [
                              [iou_rect1a_rect2a, iou_rect1a_rect2b, ... ,iou_rect1a_rect2m],
                              [iou_rect1b_rect2a, iou_rect1b_rect2b, ... ,iou_rect1b_rect2m],
                              [...]
                            ]
    """

    def iou_rects_rect(rects, ref_rect):
        """
        To calculate IoU between multiple rects and ref_rect

        :param rects: format = [[x1, y1, x2, y2],
                                [x1, y1, x2, y2],
                                [...]]
        :param ref_rect: format = [x1_gt, y1_gt, x2_gt, y2_gt]
        :return: IoUs between rects and ref_rect
        """
        x1, y1, x2, y2 = rects[:,0], rects[:,1], rects[:,2], rects[:,3]
        x1_gt, y1_gt, x2_gt, y2_gt = ref_rect

        rects_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        ref_rect_area = (x2_gt - x1_gt + 1) * (y2_gt - y1_gt + 1)

        i_x1 = xp.maximum(x1_gt, x1)
        i_y1 = xp.maximum(y1_gt, y1)
        i_x2 = xp.minimum(x2_gt, x2)
        i_y2 = xp.minimum(y2_gt, y2)

        w = xp.maximum(0.0, i_x2 - i_x1 + 1)  # if no intersection, i_x2 - i_x1 < -1, then w = 0
        h = xp.maximum(0.0, i_y2 - i_y1 + 1)  # if no intersection, i_y2 - i_y1 < -1, then h = 0
        i_area = w * h
        u_area = rects_area + ref_rect_area - i_area
        u_area[xp.where(u_area==0)] = 1e11
        iou = i_area/u_area
        return iou

    iouMat = xp.zeros((rects1.shape[0], rects2.shape[0]))
    for r, ref_rect in enumerate(rects1):
        iou = iou_rects_rect(rects2, ref_rect)
        iouMat[r] = iou
    return iouMat




def NMS(boxes, thresh=0.3, format='corner'):
    """

    :param boxes: format = [[x1, y1, x2, y2, score],
                            [x1, y1, x2, y2, score],
                            [...]]
    :param thresh: number between [0, 1]
    :return: index of remaining bbox
    """
    if(format == 'corner'):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    else:
        x1 = (boxes[:, 0] - boxes[:, 2] / 2)  # top-left x
        y1 = (boxes[:, 1] - boxes[:, 3] / 2)  # top-left y
        x2 = (boxes[:, 0] + boxes[:, 2] / 2)  # bot-right x
        y2 = (boxes[:, 1] + boxes[:, 3] / 2)  # bot-right y
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 每个boundingbox的面积
    order = scores.argsort()[::-1]  # boundingbox的置信度排序
    keep = []  # 用来保存最后留下来的boundingbox
    while order.size > 0:
        i = order[0]  # 置信度最高的boundingbox的index
        keep.append(i)  # 添加本次置信度最高的boundingbox的index

        # 当前bbox和剩下bbox之间的交叉区域
        # 选择大于x1,y1和小于x2,y2的区域
        xx1 = xp.maximum(x1[i], x1[order[1:]])
        yy1 = xp.maximum(y1[i], y1[order[1:]])
        xx2 = xp.minimum(x2[i], x2[order[1:]])
        yy2 = xp.minimum(y2[i], y2[order[1:]])

        # 当前bbox和其他剩下bbox之间交叉区域的面积
        w = xp.maximum(0.0, xx2 - xx1 + 1)
        h = xp.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 交叉区域面积 / (bbox + 某区域面积 - 交叉区域面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # 保留交集小于一定阈值的boundingbox
        inds = xp.where(iou <= thresh)[0]
        order = order[inds + 1]  # 这里的index是以order[1]为相对起始位置. 相对于order[0]的话,就在再+1

    return keep





def ctr2corner(bbox):
    bbox = xp.array(bbox)
    cx = bbox[:, 0]
    cy = bbox[:, 1]
    bw = bbox[:, 2]
    bh = bbox[:, 3]

    bbox2 = xp.empty(bbox.shape)
    bbox2[:, 0] = (bbox[:, 0] - bbox[:, 2] / 2)  # top-left x
    bbox2[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2)  # top-left y
    bbox2[:, 2] = (bbox[:, 0] + bbox[:, 2] / 2)  # bot-right x
    bbox2[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2)  # bot-right y

    return bbox2