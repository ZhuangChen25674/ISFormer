'''
数据技巧文件
'''


from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms 
import random
import torch
import numpy as np



def image_feature(img, mask, config):
    # 对图片进行技巧操作 需传入全局config
    # 返回的仍然为图片文件

    img = img.convert('RGB')

    if random.random() < 0.5 :

        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    
    base_size = config['base_size']
    crop_size = config['crop_size']

    long_size = random.randint(int(base_size*0.5), int(base_size*2.0))
    w, h = img.size
    if h > w:
        oh = long_size
        ow = int(1.0 * w * long_size / h + 0.5)
        short_size = ow
    else:
        ow = long_size
        oh = int(1.0 * h * long_size / w + 0.5)
        short_size = oh
    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
    w, h = img.size
    x1 = random.randint(0, w - crop_size)
    y1 = random.randint(0, h - crop_size)
    img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
    mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
    # gaussian blur as in PSP
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.random()))


    return img, mask





def data_feature(img, mask):
    # 数据技巧函数 返回为字典形式tensor

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    encoded_inputs = {}
    encoded_inputs['pixel_values'] = transform(img)

    tf = transforms.ToTensor() # 注意 会默认归一化
    mask = tf(mask)
    encoded_inputs['labels'] = mask

    return encoded_inputs


def new_test_feature(img, mask, config):
    # 对测试图片进行技巧操作
    # 返回的仍然为图片文件

    img = img.convert('RGB')
    mask = mask.convert('L')  
    base_size = config['crop_size']

    new_img = Image.new('RGB',(base_size,base_size),0)
    new_mask = Image.new('L',(base_size,base_size),0)
    nh = int((480-img.size[0])/2)
    nw = int((480-img.size[1])/2)
    new_img.paste(img,(nh,nw))
    new_mask.paste(mask,(nh,nw))

    return new_img, new_mask

def test_feature(img, mask, config):
    # 对测试图片进行技巧操作
    # 返回的仍然为图片文件

    img = img.convert('RGB')  
    base_size = config['base_size']

    img = img.resize((base_size, base_size), Image.BILINEAR)
    mask = mask.resize((base_size, base_size), Image.NEAREST)


    return img, mask
