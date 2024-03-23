import os
from pathlib import Path
from PIL import Image
import pandas as pd

from torchvision import transforms
from torch.utils.data import Dataset


class TrainData(Dataset):
    def __init__(self, data_root, csv_path, data_set, aug):
        super().__init__()
        anns = pd.read_csv(os.path.join(data_root, csv_path)).to_dict('records') # List of Dict
        self.img_paths = []
        self.class_ids = []
        for ann in anns:
            if ann['data set'] == data_set: # train or valid
                self.img_paths.append(os.path.join(data_root, ann['filepaths']))
                self.class_ids.append(ann['class id'])
                
        if aug:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),

                # data augmentation
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = self.transform = transforms.Compose([
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        # load image
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        # label
        class_id = self.class_ids[idx]

        return img, class_id
    

class TestData(Dataset):
    def __init__(self, img_dir):
        super().__init__()
        img_paths = Path(img_dir).glob('*.jpg')
        self.img_paths = sorted(list(img_paths))
        self.transform = transforms.Compose([
            transforms.Resize((224,224),antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, img_path.stem
    