import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TV
import torch.nn.functional as F
import torch.optim as optim
from model import YOLOV1
from dataset import PascalVOC
from utils import get_bboxes, IOU, NMS, MAP, cellboxes_to_boxes, convert_cellboxes, load_checkpoint, save_checkpoint
from loss import Loss
from torch.utils.data import Dataset, DataLoader

seed = 123

torch.manual_seed(seed)
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "yolov1.pth.tar"
EVALUATION = True
NUM_CLASS = 20
NUM_BOXES = 2
NUM_GRIDS = 7
WEIGHT_DECAY = 0
IMG_DIR = "/home/hieu/Documents/Pascal VOC/images"
LABEL_DIR = "/home/hieu/Documents/Pascal VOC/labels"

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img , boxes):
        for t in self.transforms:
            img, boxes = t(img),boxes
        return img,boxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])




def train(train_loader, model, optimizer, loss_fn, scaler):
    running_loss = []
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, (x, y) in loop:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.cuda.amp.autocast():
            preds = model(x)
            loss = loss_fn(preds, y)

        running_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"mean loss {sum(running_loss) / len(running_loss)}")

def evaluation(test_loader, model, optimizer, file):
    load_checkpoint(torch.load(file), model,optimizer)
    pred_boxes, target_boxes = get_bboxes(test_loader, model, iou_threshold=0.5, prob_threshold=0.4)
    mean_avg_pre = MAP(pred_boxes, target_boxes, iou_threshold=0.5, format="midpoints")
    print(f"Evaluation mAP: {mean_avg_pre}")


def main():
    architecture_configs = [
        (7, 64, 2, 3),  # (kernel_size, no_filters, stride, padding)
        "M",
        (3, 192, 1, 1),
        "M",
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],  # (tuple, tuple, times_of_repeat)
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",
        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),
        (3, 1024, 1, 1),
        (3, 1024, 1, 1),
    ]

    yolov1 = YOLOV1(grids = NUM_GRIDS, num_boxes= NUM_BOXES, num_classes=NUM_CLASS, architecture_config= architecture_configs).to(DEVICE)
    optimizer = optim.Adam(yolov1.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    loss = Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), yolov1,optimizer)

    train_set = PascalVOC("/home/hieu/Documents/Pascal VOC/100examples.csv",
                          IMG_DIR,
                          LABEL_DIR,
                          transform = transform)

    test_set = PascalVOC("/home/hieu/Documents/Pascal VOC/test.csv",
                          IMG_DIR,
                          LABEL_DIR,
                          transform=transform)
    train_evaluation = PascalVOC("/home/hieu/Documents/Pascal VOC/100examples.csv",
                         IMG_DIR,
                         LABEL_DIR,
                         transform=transform)

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = True)
    dev_loader = DataLoader(train_evaluation, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = True)

    if EVALUATION == False:
        for epoch in range(NUM_EPOCHS):
            pred_boxes, target_boxes = get_bboxes(dev_loader, yolov1, iou_threshold = 0.5, prob_threshold= 0.4)

            mean_avg_pre = MAP(pred_boxes,target_boxes, iou_threshold=0.5, format="midpoints")
            print(f"Train mAP: {mean_avg_pre}")
            if mean_avg_pre > 0.94:
               checkpoint = {
                   "state_dict": yolov1.state_dict(),
                   "optimizer": optimizer.state_dict(),
               }
               save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
               import time
               time.sleep(10)

            train(train_loader,yolov1, optimizer, loss,scaler)

    else: evaluation(test_loader,yolov1,optimizer,LOAD_MODEL_FILE)

main()

