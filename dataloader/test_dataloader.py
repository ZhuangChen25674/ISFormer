from torch.utils.data import Dataset
import os, operator
from PIL import Image
from .feature_extractor import test_feature, data_feature

class TestDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, config,):

        self.root_dir = config['root_dir']
        self.config = config

        
        self.img_dir = os.path.join(self.root_dir, "images", 'test')
        self.ann_dir = os.path.join(self.root_dir, "annotations", 'test')
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"
        assert operator.eq(self.images,self.annotations) #保证名称一致


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        image, segmentation_map = test_feature(image, segmentation_map, self.config)
        encoded_inputs = data_feature(image,segmentation_map)

        return encoded_inputs, image_name