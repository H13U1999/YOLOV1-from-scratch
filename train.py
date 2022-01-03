import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TV
import torch.nn.functional as F
import torch.optim as optim
from model import YOLOV1
from dataset import PascalVOC
from utils import get_bboxes, plot_image, IOU, NMS, MAP, cellboxes_to_boxes, convert_cellboxes, load_checkpoint, save_checkpoint
from loss import Loss
from torch.utils.data import Dataset, DataLoader

seed = 123

torch.manual_seed(seed)
LEARNING_RATE = 2e-5
DEVICE ="cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 240
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
NUM_CLASS = 20
WEIGHT_DECAY = 0
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self, img , boxes):
        for t in self.transforms:
            img, boxes = t(img),boxes
        return img,boxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train(train_loader, model, optimizer, loss, scaler):
    loop = tqdm(train_loader, leave =True)
    running_loss = []
    for idx, (x,y) in enumerate(loop):
        x,y = x.to(DEVICE), y.to(DEVICE)

        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = loss(pred,y)

        running_loss.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())
    print(f"mean loss {sum(running_loss)/len(running_loss)}")

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

    yolov1 = YOLOV1(grids = 7, num_boxes= 2, num_classes=3, architecture_config= architecture_configs).to(DEVICE)
    optimizer = optim.Adam(yolov1.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    loss = Loss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), yolov1,optimizer)

    train_set = PascalVOC("/home/hieu/Documents/Pascal VOC/8examples.csv",
                          "/home/hieu/Documents/Pascal VOC/images",
                          "/home/hieu/Documents/Pascal VOC/labels",
                          transform = transform)

    test_set = PascalVOC("/home/hieu/Documents/Pascal VOC/test.csv",
                          "/home/hieu/Documents/Pascal VOC/images",
                          "/home/hieu/Documents/Pascal VOC/labels",
                          transform=transform)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = False)
    test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY, shuffle = True, drop_last = True)

    for epoch in range(NUM_EPOCHS):
        pred_boxes, target_boxes = get_bboxes(train_loader, yolov1, iou_threshold = 0.5, threshold= 0.4)
        mean_avg_pre = MAP(pred_boxes,target_boxes, iou_threshold=0.5, format="midpoints")
        print(f"Train mAP: {mean_avg_pre}")
        train(train_loader,yolov1, optimizer, loss,scaler)

main()

