import glob
import torch
import numpy as np
import json
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder

def npy_loader(path):
    sample = np.load(path).astype(np.float32)
    return sample

def jpg_loader(path):
    sample = Image.open(path)
    return sample

class COCODataset(DatasetFolder):
    def __init__(self, data_path, coco_path, transform=None, labels_from = "n_annotations", ext=".png"):
        #super().__init__(data_path, transform=transform, loader = loader)
        self.transform = transform
        self.ext = ext
        if self.ext in [".png",".jpg","png","jpg"]:
            self.loader = jpg_loader
        elif self.ext in [".npy","npy"]:
            self.loader = npy_loader
        self._get_coco(coco_path)
        self.coco_path = coco_path
        self.data_path = data_path
        self.data = [(id, fn) for id, fn in zip(self.coco_images.id.values, self.coco_images.file_name.values)]
        self.labels_from = labels_from
        self._match_json_to_folder_contents()
        
    def _match_json_to_folder_contents(self):
        """if there's any discrepancy between the COCO json and folder contents, fix it"""
        ff = sorted(glob.glob(f"{self.data_path}/*{self.ext}"))
        ff = [f[f.rfind("/")+1:] for f in ff]
        imf = [fn for _, fn in self.data]
        #check that all files are in Images
        for f in ff:
            if f not in imf:
                print(f"{f} is in folder {self.data_path}, but not present in {self.coco_path}!")
        #check if there are any files in Images that aren't present in the actual folder
        bad_imid = []
        for i,fn in self.data:
            if fn not in ff:
                bad_imid.append(i)
        if len(bad_imid) != 0:
            print(f"{len(bad_imid)} images were present in {self.coco_path} but not in the folder {self.data_path}!")
            #remove those entries from Images 
            idf = self.coco_images
            self.coco_images = idf[idf['id'].isin(bad_imid)]
    
    def _get_coco(self, coco_path):
        with open(coco_path) as f:
            j = json.load(f)
        idf = pd.DataFrame(j['images'])
        adf = pd.DataFrame(j['annotations'])
        self.coco_images = idf
        self.coco_annotations = adf
        
    def _get_label(self, imid):
        anns = self.coco_annotations.where(self.coco_annotations.image_id == imid).dropna(how = 'all')
        if self.labels_from == "n_annotations": #use number of annotations per image as label
            return len(anns)
        elif self.labels_from == "category": #return most frequently occuring category in image. If equal distribution, take the one that has largest area
            cats = anns.groupby('category_id')[['category_id','area']].agg({'category_id':'first','area':'sum'}).sort_values(by='area', ascending=False)
            cat = cats.iloc[0]['category_id'] 
            return cat
        else:
            return None

    def __getitem__(self, index):
        imid, file = self.data[index]
        x = self.loader(os.path.join(self.data_path, file))

        if self.transform:
            x = self.transform(x) 
        y = self._get_label(imid)
        return x,y 
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        return f"Data shape: {self.data.shape}\nData Extrema:{np.nanmin(self.data)}, {np.nanmax(self.data)}\nData mean and std:{np.nanmean(self.data)}, {np.nanstd(self.data)}"
    
        
class RGZodDataset(COCODataset):
    def __init__(self, data_path, coco_path, transform = None, labels_from = "category", train = True):
        """RGZ dataset for object detection"""
        if data_path is None and train:
            data_path = "/path/to/train/images"
            coco_path = "/path/to/coco_annotations_train.json"
        elif data_path is None and not train:
            data_path = "path/to/test/images"
            coco_path = "/path/to/coco_annotations_test.json"
        super().__init__(data_path, coco_path, transform = transform, labels_from = labels_from)
    
class ReturnIndexDatasetRGZod(RGZodDataset):
    def __getitem__(self, idx):
        img = super(ReturnIndexDatasetRGZod, self).__getitem__(idx)
        return img, idx
    
class RGZimageDatasetClassification(COCODataset):
    def __init__(self, data_path, coco_path, transform = None, train = True, convert_boxfmt=True):
        """RGZ dataset for image classification"""
        if data_path is None and train:
            data_path = "/path/to/train_single/images"
            coco_path = "/path/to/coco_annotations_train_single.json"
        elif data_path is None and not train:
            data_path = "/path/to/test_single/images"
            coco_path = "/path/to/coco_annotations_test_single.json"
        super().__init__(data_path, coco_path, transform = transform)
        self.transform = transform
        self.convert_boxfmt = convert_boxfmt
        self.coco_annotations.rename(columns={"bbox":"boxes","category_id":"labels"}, inplace=True)
        self.label_map = {2:0,3:1,4:2,6:3,7:4,10:5} #map from labels in .yaml to true labels

    def __getitem__(self, index):
        imid, file = self.data[index]
        x = self.loader(os.path.join(self.data_path, file))
        anns = self.coco_annotations.where(self.coco_annotations.image_id == imid).dropna(how = 'all')
        #x = self.loader(os.path.join(self.data_path, file))
        if self.transform:
            x = self.transform(x) 

        target = anns.sort_values(by = "area", ascending=False).labels.iloc[0]
        target = self.label_map[int(target)]
        
        return x, target
        

class MGCLSodDataset(COCODataset):
    def __init__(self, data_path, coco_path, transform = None, labels_from = "n_annotations", train = True, ext=".npy"):
        if data_path is None and train:
            data_path = "/path/to/train"
            coco_path = "path/to/train/mgcls_coco_annotations_train.json"
        elif data_path is None and not train:
            data_path = "path/to/test/images"
            coco_path = "/path/to/test/mgcls_coco_annotations_val.json"
        super().__init__(data_path, coco_path, transform = transform, labels_from = labels_from, ext=ext)
        

