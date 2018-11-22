import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as xp
import os
import cv2
import opencvlib
from . import helper



def parse_model_config(path):
    """
    把yolov3.cfg里的各个网络层的描述都用字典表示出来, 再组织到一个列表里
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    model_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            model_defs.append({})
            model_defs[-1]['type'] = line[1:-1].rstrip() # delete ']'
            if model_defs[-1]['type'] == 'convolutional':
                model_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            model_defs[-1][key.rstrip()] = value.strip()

    hyperparams, layers_defs = model_defs[0], model_defs[1:]
    return hyperparams, layers_defs



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def downScaleBoxAttr(fmap, img_w, img_h, anchors):
    # shorthand tensor type for cuda support
    FloatTensor = torch.cuda.FloatTensor if fmap.is_cuda else torch.FloatTensor
    n_A = len(anchors)  # number of anchors in each grid cell = 3
    n_S = fmap.size(0)  # number of samples in mini-batch
    # x.size(1) is n_depth which will be processed separately
    n_H = fmap.size(2)  # feature map height
    n_W = fmap.size(3)  # feature map width
    scale_h = img_h / n_H  # downscaled factor: original image -> feature map height
    scale_w = img_w / n_W
    # 把每个anchor box缩小到feature map尺寸
    scaled_anchors = FloatTensor([(a_w / scale_w, a_h / scale_h) for a_w, a_h in anchors])
    # 把每个缩小后anchor box的长, 宽都单独收集起来
    pw = scaled_anchors[:, 0:1].view((1, n_A, 1, 1))  # pw
    ph = scaled_anchors[:, 1:2].view((1, n_A, 1, 1))  # ph
    # 以feature map的尺寸创建grid cell的左上角坐标点, 由于feature map的每个格子都是1x1大小的
    # 所以实际上就是每个feature map的格子坐标.
    grid_x = torch.arange(n_W).repeat(n_H, 1).view([1, 1, n_H, n_W]).type(FloatTensor)
    grid_y = torch.arange(n_H).repeat(n_W, 1).t().view([1, 1, n_H, n_W]).type(FloatTensor)
    return n_A, n_S, n_W, n_H, scale_w, scale_h, grid_x, grid_y, pw, ph



class BBoxProposer(nn.Module):
    """
    Propose final bbox with (nbr_bbox, bbox_attr) format, or return None when no bbox proposal
    bbox_attr = [ctrx, ctry, w, h, obj_score, cls_label]
    """
    def __init__(self, anchors, n_cls, oriImg_h, oriImg_w, objectness_thresh=0.9, nms_thresh=0.5):
        super(BBoxProposer, self).__init__()
        self.anchors = anchors
        self.n_cls = n_cls
        self.bbox_attrs = 5 + n_cls
        self.img_w = oriImg_w
        self.img_h = oriImg_h
        self.obj_thr = objectness_thresh
        self.nms_thr = nms_thresh

    def forward(self, x):
        # x.shape = [n_sample, n_depth, n_H, n_W]

        n_A, n_S, n_W, n_H, scale_w, scale_h, \
        grid_x, grid_y, pw, ph = downScaleBoxAttr(x,
                                 self.img_w,
                                 self.img_h,
                                 self.anchors)

        # (n_samples, n_anchors, 5 + n_classes, featuremap_height, featuremap_width)
        # -> (n_samples, n_anchors, featuremap_height, featuremap_width, 5 + n_classes)
        # 5 + n_classes : tx, ty, tw, th, po, classProb[0], classProb[1], ...
        prediction = x.view(n_S, n_A, self.bbox_attrs, n_W, n_H).permute(0, 1, 3, 4, 2).contiguous()

        ########################################
        # 计算当前feature map上最终预测出来的各个bbox
        ########################################
        sigma_x = torch.sigmoid(prediction[..., 0])  # σ(tx)
        sigma_y = torch.sigmoid(prediction[..., 1])  # σ(ty)
        tw = prediction[..., 2]  # tw
        th = prediction[..., 3]  # th
        pred_conf = torch.sigmoid(prediction[..., 4])  # objectness confidence score
        pred_cls = torch.argmax(torch.sigmoid(prediction[..., 5:]), dim=4)  # Cls prediction label

        # final bbox in feature map scale
        pred_boxes = xp.empty(prediction[..., :6].shape)  # placeholder

        # fill data into placeholder
        pred_boxes[..., 0] = ((sigma_x + grid_x) * scale_w).data.cpu().numpy()  # bx = (σ(tx) + Cx) * scale_w
        pred_boxes[..., 1] = ((sigma_y + grid_y) * scale_h).data.cpu().numpy()  # by = (σ(ty) + Cy) * scale_h
        pred_boxes[..., 2] = (pw * torch.exp(tw) * scale_w).data.cpu().numpy()  # bw = (pw * (e^tw)) * scale_w
        pred_boxes[..., 3] = (ph * torch.exp(th) * scale_h).data.cpu().numpy()  # bh = (ph * (e^th)) * scale_h
        pred_boxes[..., 4] = pred_conf.data.cpu().numpy()  # objectness confidence score
        pred_boxes[..., 5] = pred_cls.data.cpu().numpy()

        # remove non-object bbox
        has_object_idx = xp.where(pred_boxes[..., 4] > self.obj_thr)  # index of bbox having objects
        if(xp.size(pred_boxes[has_object_idx]) > 0):
            pred_boxes = pred_boxes[has_object_idx]
            indexes = helper.NMS(pred_boxes[..., :5], format='center', thresh=self.nms_thr)
            return pred_boxes[indexes]
        else:
            return None




class Route(nn.Module):
    """
    Route layer
    :param *prev_layer_output: previous layers' output
    """

    def __init__(self, cat_layer_index_list):
        """
        :param cat_layer_index_list: list of layer indexes to concatenate
        """
        super(Route, self).__init__()
        self.cat_layers = cat_layer_index_list

    def forward(self, *prev_layer_output):
        if(len(prev_layer_output) != len(self.cat_layers)):
            raise Exception("number of input(%0d), but expect(%0d)"
                            % (len(prev_layer_output),
                               len(self.cat_layers)))

        x = torch.cat(prev_layer_output, 1)
        return x

    def extra_repr(self):
        cat_info = "concatenate layers:"
        for i in self.cat_layers:
            cat_info = cat_info + " " + str(i) + ","
        cat_info = cat_info.rstrip(',')
        return cat_info



class ShortCut(nn.Module):
    """
    Shortcut layer
    :param *prev_layer_output: previous layers' output
    """

    def __init__(self, sum_layer_index_list):
        """
        :param sum_layer_index_list: list of layer indexes to add
        """
        super(ShortCut, self).__init__()
        self.sum_layers = sum_layer_index_list

    def forward(self, x, *prev_layer_output):
        if (len(prev_layer_output) != len(self.sum_layers)):
            raise Exception("number of input(%0d), but expect(%0d)"
                            % (len(prev_layer_output),
                               len(self.sum_layers)))

        for prev_output in prev_layer_output:
            x += prev_output
        return x

    def extra_repr(self):
        sum_info = "sum layers:"
        for i in self.sum_layers:
            sum_info = sum_info + " " + str(i) + ","
        sum_info = sum_info.rstrip(',')
        return sum_info



def create_modules(layers_defs, img_ch, img_h, img_w, anchor_bbox_iou_thr=0.5):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    anchor_bbox_iou_thr: the threshold between anchor and gtbbox such that a anchor will
                         be considered to be used in training regression
    """
    output_filters = [img_ch]  # 第一层输入图片的厚度, =3, R,G,B
    layer_list = nn.ModuleList()  # 构建一个专用的列表来存储module

    # 开始一层层构建网络
    for i, module_def in enumerate(layers_defs):
        layer = nn.Sequential()

        if module_def["type"] == "convolutional":
            # example:
            # [convolutional]
            # batch_normalize=1
            # filters=256
            # size=1
            # stride=1
            # pad=1
            # activation=leaky
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            # if kernel_size = even, pad = kernel_size/2 - 1
            # if kernel_size = odd , pad = (kernel_size-1) / 2
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            layer.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,  # 在有batchNorm的时候, bias会被batchNorm消去, 所以不用加bias
                ),
            )
            if bn:
                layer.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                layer.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))  # (padLeft=0, padRight=1, padTop=0, padBottom=1)
                layer.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def["size"]),
                stride=int(module_def["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            layer.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            # example:
            # [upsample]
            # stride=2
            upsample = nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            layer.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            # example
            # [route]
            # layers = -1, 61
            cat_layers = [int(x) for x in module_def["layers"].split(",")]  # 得到想要粘连的层的序号
            filters = sum([output_filters[layer_i] for layer_i in cat_layers])  # 把对应的层的输出在厚度上粘连起来
            layer.add_module("route_%d" % i, Route(cat_layers))  # 设置好相应大小的站位符

        elif module_def["type"] == "shortcut":
            # example:
            # [shortcut]
            # from=-3
            # activation=linear
            filters = output_filters[int(module_def["from"])]
            layer.add_module("shortcut_%d" % i, ShortCut([int(module_def["from"])]))

        elif module_def["type"] == "yolo":
            # example:
            # [yolo]
            # mask = 6,7,8
            # anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
            # classes=80
            # num=9
            # jitter=.3
            # ignore_thresh = .7
            # truth_thresh = 1
            # random=1
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]  # [6,7,8]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]  # [(10,13), (16,30), ...]
            anchors = [anchors[i] for i in anchor_idxs]  # [(116,90), (156,198), (373,326)]
            nbr_classes = int(module_def["classes"])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, nbr_classes, img_h, img_w, anchor_bbox_iou_thr)
            layer.add_module("yolo_%d" % i, yolo_layer)
        # Register module list and number of output filters
        layer_list.append(layer)
        output_filters.append(filters)

    return layer_list




class YoloSet(Dataset):
    """
    :return : - img : the processed ready to be used in training based on indicating size
              - labels : bbox info in (cls_id, ctr_x, ctr_y, box_w, box_y) format
    """
    def __init__(self, path_file, img_h=416, img_w=416):
        with open(path_file, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('COCO/', 'COCO/labels/').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_wh = (img_w, img_h)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        ##########################################
        # process input image to be ready for training
        ##########################################
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = cv2.imread(img_path)  # (H, W, C)
        while(len(img.shape) < 3):
            index = (index+1) % len(self.img_files)
            img = cv2.imread(img_path)  # (H, W, C)

        h, w = img.shape[0:2]
        dim_diff = xp.abs(h - w)
        # 计算长宽差，根据差值进行padding补齐成正方图
        # 如果长宽差为偶数=2k, 左/上, 右/下各补一半k
        # 如果长宽差为奇数=2k+1, 左/上补k, 右/下补k+1
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # 胖图上下补, 瘦图左右补, 以值=128进行填充, 并scale到[0, 1]之间
        padding = (pad1, pad2, 0, 0) if (h <= w) else (0, 0, pad1, pad2)
        img = cv2.copyMakeBorder(img, *padding, cv2.BORDER_CONSTANT, value=[128] * 3)
        img = img / 255

        padded_h, padded_w, _ = img.shape
        # Resize
        img = opencvlib.Resize(img, *self.img_wh)
        # Channels-first
        img = xp.transpose(img, (2, 0, 1))  # (C, H, W)

        ##########################################
        # process label to be ready for training
        ##########################################
        label_path = self.label_files[index % len(self.label_files)].rstrip()
        if os.path.exists(label_path):
            # reshape to only 5 columns
            # clsid  center_x  center_y     w        h
            # 45     0.479492  0.688771  0.955609  0.595500
            # 45     0.736516  0.247188  0.498875  0.476417
            # 50     0.637063  0.732938  0.494125  0.510583
            labels = xp.loadtxt(label_path).reshape(-1, 5)

            ctr_x = ((labels[:, 1] * w).astype(xp.int32) + padding[2]).astype(xp.float64)
            ctr_y = ((labels[:, 2] * h).astype(xp.int32) + padding[0]).astype(xp.float64)
            bw = ((labels[:, 3] * w).astype(xp.int32)).astype(xp.float64)
            bh = ((labels[:, 4] * h).astype(xp.int32)).astype(xp.float64)

            ctr_x /= padded_w
            ctr_y /= padded_h
            bw /= padded_w
            bh /= padded_h

            labels[:, 1] = ctr_x  # Ctr_x in padded image size
            labels[:, 2] = ctr_y  # Ctr_y
            labels[:, 3] = bw  # bw
            labels[:, 4] = bh  # bh

        return img, labels





class YOLOLayer(nn.Module):
    """
    Detection/Train layer
    when set model to eval() mode, detection function is used
    when set model to train() mode, training function is used
    """

    def __init__(self, anchors, n_cls, oriImg_h, oriImg_w, anchor_bbox_iou_thr):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.n_cls = n_cls
        self.oriImg_h = oriImg_h
        self.oriImg_w = oriImg_w
        self.bboxProposer = BBoxProposer(self.anchors,
                                         self.n_cls,
                                         self.oriImg_h,
                                         self.oriImg_w)
        self.anchor_bbox_iou_thr = anchor_bbox_iou_thr

        self.mse_loss = nn.MSELoss()  # Coordinate loss
        self.bce_loss = nn.BCELoss()  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, labels=None):
        is_training = labels is not None
        if(is_training):
            #print("In train mode")
            n_A, n_S, n_W, n_H, scale_w, scale_h, \
            grid_x, grid_y, pw, ph = downScaleBoxAttr(x,
                                                      self.oriImg_w,
                                                      self.oriImg_h,
                                                      self.anchors)

            # x.shape = (n_samples, n_anchors, featuremap_height, featuremap_width, 5 + n_classes)
            x = x.view(n_S, n_A, n_H, n_W, -1)

            # down scaled anchors for an image
            FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
            self.mse_loss = self.mse_loss.cuda() if x.is_cuda else self.mse_loss

            pw = pw.repeat(n_S, 1, n_H, n_W).view(n_A, n_H, n_W).type(FloatTensor)
            ph = ph.repeat(n_S, 1, n_H, n_W).view(n_A, n_H, n_W).type(FloatTensor)
            grid_x = grid_x.repeat(n_S, n_A, 1, 1).view(n_A, n_H, n_W).type(FloatTensor)
            grid_y = grid_y.repeat(n_S, n_A, 1, 1).view(n_A, n_H, n_W).type(FloatTensor)
            grid_x.require_grad = False
            grid_y.require_grad = False

            # x1, x2, y1, y2 in anchors in all aspect ratios
            # x1 format = (n_aspect_ratio, famp_h, fmap_w)
            anchor_allRatio_x1 = (grid_x - pw/2).clamp(min=0, max=n_W - 1)
            anchor_allRatio_y1 = (grid_y - ph/2).clamp(min=0, max=n_H - 1)
            anchor_allRatio_x2 = (grid_x + pw/2).clamp(min=0, max=n_W - 1)
            anchor_allRatio_y2 = (grid_y + ph/2).clamp(min=0, max=n_H - 1)

            # rearrange anchors into format of (n_aspect_ratio, fmap_h, fmap_w, (x1,y1,x2,y2))
            anchor_x1y1x2y2 = FloatTensor(n_A, n_H, n_W, 4)
            for idx in range(n_A):
                anchor_x1y1x2y2[idx, ...] = torch.cat(
                                            (anchor_allRatio_x1[idx, ...].unsqueeze(0),  # extend one dimension
                                             anchor_allRatio_y1[idx, ...].unsqueeze(0),
                                             anchor_allRatio_x2[idx, ...].unsqueeze(0),
                                             anchor_allRatio_y2[idx, ...].unsqueeze(0)
                                             ), 0).permute(1, 2, 0).contiguous()

            loss_batch = 0
            for img_idx in range(n_S):
                n_bbox = labels.size(1)
                gtbbox_clsid = labels[img_idx, :, 0].view(n_bbox, -1).type(LongTensor)
                one_hot_clsid = torch.zeros(n_bbox, self.n_cls).type(LongTensor).scatter_(1, gtbbox_clsid, 1)
                # rearrange bbox into format of (n_bbox, (x1,y1,x2,y2))
                bbox_x1y1x2y2 = FloatTensor(labels.size(1), 4)
                bbox_x1y1x2y2[..., 0] = n_W * (labels[img_idx, :, 1] - labels[img_idx, :, 3] / 2)  # top-left x in fmap
                bbox_x1y1x2y2[..., 1] = n_H * (labels[img_idx, :, 2] - labels[img_idx, :, 4] / 2)  # top-left y in fmap
                bbox_x1y1x2y2[..., 2] = n_W * (labels[img_idx, :, 1] + labels[img_idx, :, 3] / 2)  # bot-right x in fmap
                bbox_x1y1x2y2[..., 3] = n_H * (labels[img_idx, :, 2] + labels[img_idx, :, 4] / 2)  # bot-right y in fmap

                # x.shape = (n_samples, n_anchors, featuremap_height, featuremap_width, 5 + n_classes)
                train_mask = xp.zeros(x.shape[1:])  # mask for one image
                ground_truth = xp.zeros(x.shape[1:])  # ground truth for one image
                for aspect_idx in range(anchor_x1y1x2y2.size(0)):
                    anchors = anchor_x1y1x2y2[aspect_idx, ...].view(-1, 4)
                    iou_mat = helper.CalcIoU(anchors.cpu().numpy(), bbox_x1y1x2y2.cpu().numpy())
                    keep_idx = xp.where(iou_mat >= self.anchor_bbox_iou_thr)
                    keep_anchor_idx = keep_idx[0]
                    # anchor_mask
                    # [
                    #  [is_anchor_valid, target_bbox_idx]
                    #  [is_anchor_valid, target_bbox_idx]
                    #  [is_anchor_valid, target_bbox_idx]
                    #  [is_anchor_valid, target_bbox_idx]
                    #  ...
                    # ]
                    gt_val = xp.zeros((n_H*n_W, 5+self.n_cls))
                    gt_val[keep_anchor_idx, 4] = 1  # objectness score
                    train_mask[aspect_idx, keep_anchor_idx // n_W, keep_anchor_idx % n_W, :] = 1  # set mask
                    anchor_target_bbox_idx = xp.argmax(iou_mat, axis=1)
                    gt_val[:, 0] = anchor_target_bbox_idx

                    for anch_idx in keep_anchor_idx:
                        gtbbox_idx = int(gt_val[anch_idx, 0])
                        gtbbox_attr = labels[img_idx, gtbbox_idx, 1:5].cpu().numpy()
                        gt_val[anch_idx, :4] = gtbbox_attr
                        gt_val[anch_idx, 5:] = one_hot_clsid[gtbbox_idx, :].cpu().numpy()

                    ground_truth[aspect_idx, ...] = gt_val.reshape(n_H, n_W, -1)

                train_mask = FloatTensor(train_mask)
                ground_truth = FloatTensor(ground_truth)
                train_mask.require_grad = False
                ground_truth.require_grad = False

                fmap_single_img = x[img_idx, ...]

                fmap_single_img[..., 0] = (torch.sigmoid(fmap_single_img[..., 0]) + grid_x) * scale_w
                fmap_single_img[..., 1] = (torch.sigmoid(fmap_single_img[..., 1]) + grid_y) * scale_h  # by = (σ(ty) + Cy) * scale_h
                fmap_single_img[..., 2] = pw * torch.exp(fmap_single_img[..., 2]) * scale_w  # bw = (pw * (e^tw)) * scale_w
                fmap_single_img[..., 3] = ph * torch.exp(fmap_single_img[..., 3]) * scale_h  # bh = (ph * (e^th)) * scale_h
                fmap_single_img[..., 4] = torch.sigmoid(fmap_single_img[..., 4])
                fmap_single_img[..., 5:] = torch.sigmoid(fmap_single_img[..., 5:])  # Cls prediction label

                loss_single_img = self.mse_loss(fmap_single_img*train_mask, ground_truth*train_mask)  # remove non-object predictions
                loss_batch += loss_single_img

            loss_batch /= n_S
            #print("loss_batch = ", loss_batch)
            return loss_batch

        else:
            #print("In eval mode")
            return self.bboxProposer(x)







class Darknet(nn.Module):

    def __init__(self, config_path):
        super(Darknet, self).__init__()
        hyperparams, self.module_defs = parse_model_config(config_path)
        self.module_list = create_modules(self.module_defs,
                                          int(hyperparams["channels"]),
                                          int(hyperparams["height"]),
                                          int(hyperparams["width"]),
                                          0.5)


    def forward(self, x, labels=None):
        """
        :param x: format = (N, C, H, W)
        :param labels: bbox info
        :return: bboxes or losses
        """
        is_training = (labels is not None) and self.training
        yolo_bbox = xp.empty((1, 6), xp.float64)
        yolo_loss = 0
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_def["type"] == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                x = module[0](*[layer_outputs[i] for i in layer_i])
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = module[0](x, layer_outputs[layer_i])
            elif module_def["type"] == "yolo":
                # Train phase: get loss
                if is_training:
                    # x now is loss
                    x = module[0](x, labels)
                    yolo_loss += x
                else:
                    # x now is final bbox prediction
                    x = module(x)
                    if x is not None:
                        yolo_bbox = xp.concatenate((yolo_bbox, x), axis=0)

            # collect output of last layer
            layer_outputs.append(x)

        if not is_training:
            return yolo_bbox[1:, :]
        else:
            return yolo_loss


    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""
        fp = open(weights_path, "rb")
        header = xp.fromfile(fp, dtype=xp.int32, count=5)  # First five are header values

        # Needed to write header when saving weights
        self.header_info = header

        self.seen = header[3]
        weights = xp.fromfile(fp, dtype=xp.float32)  # The rest are weights
        fp.close()

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


    def save_weights(self, path, cutoff=-1):
        """
        :param path    - path of the new weights file
        :param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)
        """
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()


















def load_classes(path='./coco.names'):
    """
    从文件里读取所有类名
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names



# def show(tarSet):
#     names = load_classes()
#     for img, labels in tarSet:
#         img = xp.transpose(img, (1,2,0))
#         h, w, _ = img.shape
#         ids = labels[:, 0]
#         x1 = w * (labels[:, 1] - labels[:, 3] / 2)  # top-left x in original image size
#         y1 = h * (labels[:, 2] - labels[:, 4] / 2)  # top-left y in original image size
#         x2 = w * (labels[:, 1] + labels[:, 3] / 2)  # bot-right x in original image size
#         y2 = h * (labels[:, 2] + labels[:, 4] / 2)  # bot-right y in original image size
#
#         x1 = x1.astype(xp.int32)
#         y1 = y1.astype(xp.int32)
#         x2 = x2.astype(xp.int32)
#         y2 = y2.astype(xp.int32)
#
#         colors = [opencvlib.COLOR_BLUE, opencvlib.COLOR_GREEN, opencvlib.COLOR_RED]
#         import random
#         print("---------------------")
#         for a,b,c,d,id in zip(x1,y1,x2,y2,ids):
#             name = names[int(id)]
#             color = random.choice(colors)
#             cv2.rectangle(img, (a, b), (c, d), color, 2)
#             t_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
#             cv2.putText(img, name, (a, b + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
#
#         cv2.imshow("ori", img)
#
#         opencvlib.WaitEscToExit()













# model = Darknet("./yolov3.cfg")
# model.load_weights("/home/qs/opt/yolo_v3/yolov3.weights")
# img = cv2.imread("../samples/giraffe.jpg")
# img = opencvlib.Resize(img, 416, 416)
# ori = img
# img = xp.transpose(img, (2, 0, 1))
# c, h, w = img.shape
# img = img.reshape(1, c, h, w)
# img = torch.FloatTensor(img)
#
#
# pred, _ = model(img)
#
# cx = pred[:, 0]
# cy = pred[:, 1]
# bw = pred[:, 2]
# bh = pred[:, 3]
#
# x1 = (cx - bw / 2).astype(xp.int32)  # top-left x in original image size
# y1 = (cy - bh / 2).astype(xp.int32)  # top-left y in original image size
# x2 = (cx + bw / 2).astype(xp.int32)  # bot-right x in original image size
# y2 = (cy + bh / 2).astype(xp.int32)  # bot-right y in original image size
#
# img = ori
# img = opencvlib.Resize(img, 416, 416)
#
#
# for a,b,c,d in zip(x1,y1,x2,y2):
#     cv2.rectangle(img, (a, b), (c, d), opencvlib.COLOR_BLUE, 2)
#
# cv2.imshow("ori", img)
# opencvlib.WaitEscToExit()


