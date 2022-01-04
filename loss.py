import torch
import torch.nn as nn
from utils import IOU

class Loss(nn.Module):
    def __init__(self,grids = 7, num_boxes=2, num_classes=20):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction = "sum") #YOLO do sum
        self.grids = grids
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self,prediction,target):
        prediction = prediction.reshape(-1, self.grids, self.grids,self.num_boxes*5 + self.num_classes)
        incre = 0
        ious = 0
        for _ in range(self.num_boxes):
            if incre == 0:
                ious = IOU(prediction[..., self.num_classes+1+incre:self.num_classes+5+incre], target[..., self.num_classes+1+incre: self.num_classes+5+incre])
            elif incre != 0:
                bb = IOU(prediction[..., self.num_classes+1+incre:self.num_classes+5+incre], target[..., self.num_classes+1+incre: self.num_classes+5+incre])
                ious = torch.cat([ious.unsqueeze(0), bb.unsqueeze(0)], dim=0)
            incre += 5

        ious_maxes, best_boxes = torch.max(ious,dim = 0) #position 1 or 0 if bb1 max => 0 if bb2 max => 1
        exists_boxes = target[..., 20].unsqueeze(3) # is there an obj

        #coord loss
        boxes_predictions = exists_boxes * (best_boxes*prediction[..., 26:30] + (1-best_boxes) * prediction[..., 21:25]) # which box is responsible for prediction
        boxes_targets = exists_boxes * target[..., 21:25]
        boxes_predictions[...,2:4] = torch.sign(boxes_predictions[...,2:4]) * torch.sqrt(torch.abs(boxes_predictions[...,2:4] +1e-6)) # height and width
        # target (Batch, grids,grids, 4)
        boxes_targets[...,2:4] = torch.sqrt(boxes_targets[...,2:4])

        #flatten (Batch, grids,grids, 4) to (Batch* grids * grids, 4) for computing the loss of coordinates
        coord_loss = self.mse(torch.flatten(boxes_predictions, end_dim=-2),
                              torch.flatten(boxes_targets, end_dim = -2))


        #object loss
        boxes_obj_predictions = exists_boxes * (
                    best_boxes * prediction[..., 25:26] + (1 - best_boxes) * prediction[..., 20:21])
        boxes_obj_targets = exists_boxes * target[..., 20:21]
        #same as the coordinate loss
        object_loss = self.mse(torch.flatten(boxes_obj_predictions),
                               torch.flatten(boxes_obj_targets)) #(batch*grids*grids)


        # no object loss (Batch, grids,grids, 1) = > (Batch, grids *grids)
        not_exists_boxes = 1 - exists_boxes
        boxes_noobj_prediction1 = not_exists_boxes * (prediction[..., 20:21])
        boxes_noobj_prediction2 = not_exists_boxes * (prediction[..., 25:26])
        boxes_noobj_targets = not_exists_boxes * target[..., 20:21]
        no_object_loss = self.mse(torch.flatten(boxes_noobj_prediction1, start_dim = 1),
                                  torch.flatten(boxes_noobj_targets, start_dim = 1))
        no_object_loss += self.mse(torch.flatten(boxes_noobj_prediction2, start_dim=1),
                                  torch.flatten(boxes_noobj_targets, start_dim=1))


        #classification loss (Batch, grids,grids, 1) => (Batch * grids *grids,20)
        boxes_class_predictions =  exists_boxes * prediction[..., :20]
        boxes_class_targets = exists_boxes * target[..., :20]
        class_loss = self.mse(torch.flatten(boxes_class_predictions, end_dim=-2),
                              torch.flatten(boxes_class_targets, end_dim = -2))

        loss = (self.lambda_coord * coord_loss
                + object_loss
                + self.lambda_noobj * no_object_loss
                + class_loss)

        return loss


"""
Implementation of Yolo Loss Function from the original yolo paper

"""

# import torch
# import torch.nn as nn
# from utils import IOU
#
#
# class Loss(nn.Module):
#     """
#     Calculate the loss for yolo (v1) model
#     """
#
#     def __init__(self, S=7, B=2, C=20):
#         super(Loss, self).__init__()
#         self.mse = nn.MSELoss(reduction="sum")
#
#         """
#         S is split size of image (in paper 7),
#         B is number of boxes (in paper 2),
#         C is number of classes (in paper and VOC dataset is 20),
#         """
#         self.S = S
#         self.B = B
#         self.C = C
#
#         # These are from Yolo paper, signifying how much we should
#         # pay loss for no object (noobj) and the box coordinates (coord)
#         self.lambda_noobj = 0.5
#         self.lambda_coord = 5
#
#     def forward(self, predictions, target):
#         # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
#         predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
#
#         # Calculate IoU for the two predicted bounding boxes with target bbox
#         iou_b1 = IOU(predictions[..., 21:25], target[..., 21:25])
#         iou_b2 = IOU(predictions[..., 26:30], target[..., 21:25])
#         ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
#
#         # Take the box with highest IoU out of the two prediction
#         # Note that bestbox will be indices of 0, 1 for which bbox was best
#         iou_maxes, bestbox = torch.max(ious, dim=0)
#         exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i
#
#         # ======================== #
#         #   FOR BOX COORDINATES    #
#         # ======================== #
#
#         # Set boxes with no object in them to 0. We only take out one of the two
#         # predictions, which is the one with highest Iou calculated previously.
#         box_predictions = exists_box * (
#             (
#                 bestbox * predictions[..., 26:30]
#                 + (1 - bestbox) * predictions[..., 21:25]
#             )
#         )
#
#         box_targets = exists_box * target[..., 21:25]
#
#         # Take sqrt of width, height of boxes to ensure that
#         box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
#             torch.abs(box_predictions[..., 2:4] + 1e-6)
#         )
#         box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
#
#         box_loss = self.mse(
#             torch.flatten(box_predictions, end_dim=-2),
#             torch.flatten(box_targets, end_dim=-2),
#         )
#
#         # ==================== #
#         #   FOR OBJECT LOSS    #
#         # ==================== #
#
#         # pred_box is the confidence score for the bbox with highest IoU
#         pred_box = (
#             bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
#         )
#
#         object_loss = self.mse(
#             torch.flatten(exists_box * pred_box),
#             torch.flatten(exists_box * target[..., 20:21]),
#         )
#
#         # ======================= #
#         #   FOR NO OBJECT LOSS    #
#         # ======================= #
#
#         #max_no_obj = torch.max(predictions[..., 20:21], predictions[..., 25:26])
#         #no_object_loss = self.mse(
#         #    torch.flatten((1 - exists_box) * max_no_obj, start_dim=1),
#         #    torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
#         #)
#
#         no_object_loss = self.mse(
#             torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
#             torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
#         )
#
#         no_object_loss += self.mse(
#             torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
#             torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
#         )
#
#         # ================== #
#         #   FOR CLASS LOSS   #
#         # ================== #
#
#         class_loss = self.mse(
#             torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
#             torch.flatten(exists_box * target[..., :20], end_dim=-2,),
#         )
#
#         loss = (
#             self.lambda_coord * box_loss  # first two rows in paper
#             + object_loss  # third row in paper
#             + self.lambda_noobj * no_object_loss  # forth row
#             + class_loss  # fifth row
#         )
#
#         return loss
