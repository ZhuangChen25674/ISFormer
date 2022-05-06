import numpy as np
from model.new_segformer import Segformer
from dataloader.test_dataloader import TestDataset
from torch.utils.data import DataLoader
from configs import CONFIG
from datetime import datetime
import torch
from torch import nn
from tqdm import tqdm
import os 
from PIL import Image
from utils import utils

dmy = datetime.now()
dmy_string = dmy.strftime("%d-%m-%Y")

device = torch.device("cpu")
checkpoint = torch.load(CONFIG['load_checkpoint'],map_location=device)
print('device is {}'.format(device))
model = Segformer(**CONFIG['SegformerConfig'])
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
metrics = CONFIG['metrics']

test_dataset = TestDataset(CONFIG)

test_dataloader = DataLoader(test_dataset, batch_size=1, \
                                num_workers=CONFIG['num_workers'])

res_iou = np.array([])
res_niou = np.array([])

path = os.path.join(CONFIG['pred_dir'],dmy_string)
if not os.path.exists(path) :
    #pred/dmy/res
    os.makedirs(os.path.join(path,'res'))

model.eval()
with torch.no_grad():

    for batch, img_name in tqdm(test_dataloader):

        metrics['iou_metric'].reset()
        metrics['nIoU_metric'].reset()

        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values)
        predicted = nn.functional.interpolate(outputs, size=labels.shape[-2:], mode="bilinear", align_corners=False)

        metrics['iou_metric'].update(predicted,labels)
        metrics['nIoU_metric'].update(predicted,labels)
        _, IoU = metrics['iou_metric'].get()
        _, nIoU = metrics['nIoU_metric'].get()

        res_iou = np.append(res_iou,IoU)
        res_niou = np.append(res_niou,nIoU)

        # outputs = (outputs.squeeze().numpy() > 0).astype('uint8')
        # outputs *= 255

        # img = Image.fromarray(outputs)
        # img = img.resize((480,480), Image.NEAREST)
        # img = utils.ChangeColor(img)

        # label = labels.squeeze().numpy().astype('uint8')
        # label *= 255
        # label = Image.fromarray(label)
        # label = utils.ChangeColor(label)

        # # pred/dmy/.log
        log_path = os.path.join(path,'{}.log'.format(dmy_string))
        with open(log_path,'a') as f:
            f.write('{} IoU is {:.4f} nIou is {:.4f}\n'.format(img_name[0], IoU,nIoU))

        # img.save(os.path.join(path,'res',img_name[0]))
        # label.save('/home/tcs1/data2/2022-cz/old-Segformer/pred/427/label_480/{}'.format(img_name[0]))




log_path = os.path.join(path,'{}.log'.format(dmy_string))
with open(log_path,'a') as f:
    print(len(res_iou))
    f.write('average IoU is {:.4f}\n'.format(res_iou.mean()))
    f.write('average nIoU is {:.4f}\n'.format(res_niou.mean()))
    f.write('Finished\n')


  