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
                ious = IOU(prediction[..., self.num_classes+1+incre:self.num_classes+5+incre], target[..., self.num_classes+1+incre: self.num_classes+5+incre]).unsqueeze(0)
            elif incre != 0:
                bb = IOU(prediction[..., self.num_classes+1+incre:self.num_classes+5+incre], target[..., self.num_classes+1+incre: self.num_classes+5+incre])
                ious = torch.cat([ious, bb.unsqueeze(0)], dim=0)
            incre += 5

        ious_maxes, best_boxes = torch.max(ious,dim = 0) #position 1 or 0 if bb1 max => 0 if bb2 max => 1
        exists_boxes = target[...,20].unsqueeze(3) # is there an obj

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
        boxes_noobj_prediction1 = not_exists_boxes * (prediction[...,20:21])
        boxes_noobj_prediction2 = not_exists_boxes * (prediction[..., 25:26])
        boxes_noobj_targets = not_exists_boxes * target[...,20:21]
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
                + self.lambda_noobj*no_object_loss
                + class_loss)
        return loss