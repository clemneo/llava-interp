import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Tuple
from pycocotools.coco import COCO 
from PIL import Image

class COCOImageDataset(Dataset):
    def __init__(self, data_dir: str, data_type:str, ann_file: str, transform=None, filter_fn: Callable[[Dict], bool] = None):
        self.data_dir = data_dir
        self.data_type = data_type
        self.coco = COCO(ann_file)
        self.transform = transform

        self.filter_fn = filter_fn
        self.filtered_img_ids = self._filter_images()
        
    def _filter_images(self) -> List[int]:
        filtered_ids = []

        if self.filter_fn is None:
            filtered_ids = self.coco.getImgIds()
            print("No filter passed! All images will be in the dataset.")
            return filtered_ids

        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            if self.filter_fn(anns):
                filtered_ids.append(img_id)

        print(f"{len(filtered_ids)}/{len(self.coco.getImgIds())} images passed the filter.")

        return filtered_ids
    
    def __len__(self) -> int:
        return len(self.filtered_img_ids)
         
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        img_id = self.filtered_img_ids[idx]

        img_info = self.coco.loadImgs(img_id)[0]
        img = Image.open(f"{self.data_dir}/{self.data_type}/{img_info['file_name']}")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if self.transform:
            img = self.transform(img)        

        return img_id, img, anns