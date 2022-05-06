from torch.utils.data import Dataset
import os, operator
from PIL import Image
from .feature_extractor import image_feature, data_feature,test_feature

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, config, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = config['root_dir']
        # self.feature_extractor = feature_extractor
        self.train = train
        self.config = config

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        # print(self.img_dir)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)
        
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
        
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        # encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")
        
        image, segmentation_map = test_feature(image, segmentation_map, self.config)
        encoded_inputs = data_feature(image,segmentation_map)

        return encoded_inputs