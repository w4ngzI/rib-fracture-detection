import os 
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from itertools import product
from skimage.measure import regionprops


class TrainDataset(Dataset):

    def __init__(self, image_dir, label_dir, crop_size = 64, num_samples = 4, transforms = None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.id_list = sorted([x.split("-")[0] for x in os.listdir(image_dir)])
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.transforms = transforms

    def __len__(self):
        return len(self.id_list)
    
    def get_pos_centroids(self, label_arr):
        centroids = [tuple([round(x + np.random.uniform(-20, 20)) for x in prop.centroid]) for prop in regionprops(label_arr)]
        
        return centroids

    def get_sym_centroids(self, pos_centroids, width):
        sym_neg_centroids = [(width - x, y, z) for x, y, z in pos_centroids]

        return sym_neg_centroids

    def get_spine_centroids(self, im_info, crop_size, num_samples):
        '''
            zmin, zmax can be modified since it seems that the neck part is not included in this range\
            namely, to avoid choose the neck part
        '''
        xmin, xmax = im_info[0] // 2 - 40, im_info[0] // 2 + 40
        ymin, ymax = 300, 400
        zmin, zmax = crop_size // 2, im_info[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(xmin, xmax),
            np.random.randint(ymin, ymax),
            np.random.randint(zmin, zmax),
        ) for i in range(num_samples)]

        return spine_neg_centroids 

    def get_neg_centroids(self, pos_centroids, im_info):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self.get_sym_centroids(pos_centroids, im_info[0])
        #print(num_pos)   2 in label_1
        
        if num_pos < self.num_samples // 2:
            spine_centroids = self.get_spine_centroids(im_info, self.crop_size, self.num_samples - num_pos * 2)
        else:
            spine_centroids = self.get_spine_centroids(im_info, self.crop_size, num_pos)
        
        return sym_neg_centroids + spine_centroids

    def get_roi_centroids(self, label_arr):
        pos_centroids = self.get_pos_centroids(label_arr)
        neg_centroids = self.get_neg_centroids(pos_centroids, label_arr.shape)

        num_pos = len(pos_centroids)
        num_neg = len(neg_centroids)

        # number of positive instances is more than 4, then positive == negtive == 2
        if num_pos >= self.num_samples:   
            num_pos = self.num_samples // 2
            num_neg = self.num_samples // 2
        # number of positive instanves is in the range(2, 4), then keep all the positive ones
        elif num_pos >= self.num_samples // 2:
            num_neg = self.num_samples - num_pos

        #if the number of instances are more than the kept number, randomly choose some of them 
        if num_pos < len(pos_centroids):
            pos_centroids = [pos_centroids[i] for i in np.random.choice(range(0, len(pos_centroids)), size = num_pos, replace = False)]
        if num_neg < len(neg_centroids):
            neg_centroids = [neg_centroids[i] for i in np.random.choice(range(0, len(neg_centroids)), size = num_neg, replace = False)]

        roi_centroids = pos_centroids + neg_centroids

        return roi_centroids

    def crop_roi(self, data, centroid):
        #it seems that the value of the areas, which are neigher background nor bones, is around -1024
        #in other words, the pixel value of other organ is around -1024  
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        begin = [max(0, centroid[i] - self.crop_size // 2) for i in range(len(centroid))]
        end = [min(data.shape[i], centroid[i] + self.crop_size // 2) for i in range(len(centroid))]
        roi_begin = [max(0, self.crop_size // 2 - centroid[i]) for i in range(len(centroid))]
        roi_end = [min(data.shape[i] - (centroid[i] - self.crop_size // 2), self.crop_size) for i in range(len(centroid))]
        
        roi[roi_begin[0]:roi_end[0], roi_begin[1]:roi_end[1], roi_begin[2]:roi_end[2]] = \
            data[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]

        return roi

    def apply_transforms(self, image_roi):
        for t in self.transforms:
            image_roi = t(image_roi)
            print(image_roi.shape)
        return image_roi

    def __getitem__(self, index):
        id_ = self.id_list[index]
        image_path = os.path.join(self.image_dir, f"{id_}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{id_}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)

        image_arr = image.get_fdata().astype(np.float)
        label_arr = label.get_fdata().astype(np.uint8)
        roi_centroids = self.get_roi_centroids(label_arr)

        image_rois = [self.crop_roi(image_arr, centroid) for centroid in roi_centroids]
        label_rois = [self.crop_roi(label_arr, centroid) for centroid in roi_centroids]

        if self.transforms is not None:
            image_rois = [self.apply_transforms(image_roi) for image_roi in image_rois]
        #concatenate in the first dimension, then add a new axis
        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis], dtype = torch.float)
        #change 1, 2 in label into 1 by bool operation
        label_rois = (np.stack(label_rois) > 0).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis], dtype = torch.float)


        return image_rois, label_rois

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        return image_rois, label_rois

class ValDataset(Dataset):
    #get_spine_centroids need to be modified

    def __init__(self, image_dir, label_dir, crop_size = 64, num_samples = 4, transforms = None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.id_list = sorted([x.split("-")[0] for x in os.listdir(image_dir)])
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.transforms = transforms

    def __len__(self):
        return len(self.id_list)
    
    def get_pos_centroids(self, label_arr):
        centroids = [tuple([round(x + np.random.uniform(-20, 20)) for x in prop.centroid]) for prop in regionprops(label_arr)]
        
        return centroids

    def get_sym_centroids(self, pos_centroids, width):
        sym_neg_centroids = [(width - x, y, z) for x, y, z in pos_centroids]

        return sym_neg_centroids

    def get_spine_centroids(self, im_info, crop_size, num_samples):
        '''
            zmin, zmax can be modified since it seems that the neck part is not included in this range\
            namely, to avoid choose the neck part
        '''
        xmin, xmax = im_info[0] // 2 - 40, im_info[0] // 2 + 40
        ymin, ymax = 300, 400
        zmin, zmax = crop_size // 2, im_info[2] - crop_size // 2
        spine_neg_centroids = [(
            np.random.randint(xmin, xmax),
            np.random.randint(ymin, ymax),
            np.random.randint(zmin, zmax),
        ) for i in range(num_samples)]

        return spine_neg_centroids 

    def get_neg_centroids(self, pos_centroids, im_info):
        num_pos = len(pos_centroids)
        sym_neg_centroids = self.get_sym_centroids(pos_centroids, im_info[0])
        #print(num_pos)   2 in label_1
        
        if num_pos < self.num_samples // 2:
            spine_centroids = self.get_spine_centroids(im_info, self.crop_size, self.num_samples - num_pos * 2)
        else:
            spine_centroids = self.get_spine_centroids(im_info, self.crop_size, num_pos)
        
        return sym_neg_centroids + spine_centroids

    def get_roi_centroids(self, label_arr):
        pos_centroids = self.get_pos_centroids(label_arr)
        neg_centroids = self.get_neg_centroids(pos_centroids, label_arr.shape)

        num_pos = len(pos_centroids)
        num_neg = len(neg_centroids)

        # number of positive instances is more than 4, then positive == negtive == 2
        if num_pos >= self.num_samples:   
            num_pos = self.num_samples // 2
            num_neg = self.num_samples // 2
        # number of positive instanves is in the range(2, 4), then keep all the positive ones
        elif num_pos >= self.num_samples // 2:
            num_neg = self.num_samples - num_pos

        #if the number of instances are more than the kept number, randomly choose some of them 
        if num_pos < len(pos_centroids):
            pos_centroids = [pos_centroids[i] for i in np.random.choice(range(0, len(pos_centroids)), size = num_pos, replace = False)]
        if num_neg < len(neg_centroids):
            neg_centroids = [neg_centroids[i] for i in np.random.choice(range(0, len(neg_centroids)), size = num_neg, replace = False)]

        roi_centroids = pos_centroids + neg_centroids

        return roi_centroids

    def crop_roi(self, data, centroid):
        #it seems that the value of the areas, which are neigher background nor bones, is around -1024
        #in other words, the pixel value of other organ is around -1024  
        roi = np.ones(tuple([self.crop_size] * 3)) * (-1024)

        begin = [max(0, centroid[i] - self.crop_size // 2) for i in range(len(centroid))]
        end = [min(data.shape[i], centroid[i] + self.crop_size // 2) for i in range(len(centroid))]
        roi_begin = [max(0, self.crop_size // 2 - centroid[i]) for i in range(len(centroid))]
        roi_end = [min(data.shape[i] - (centroid[i] - self.crop_size // 2), self.crop_size) for i in range(len(centroid))]
        
        roi[roi_begin[0]:roi_end[0], roi_begin[1]:roi_end[1], roi_begin[2]:roi_end[2]] = \
            data[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2]]

        return roi

    def apply_transforms(self, image_roi):
        for t in self.transforms:
            image_roi = t(image_roi)
            print(image_roi.shape)
        return image_roi

    def __getitem__(self, index):
        id_ = self.id_list[index]
        image_path = os.path.join(self.image_dir, f"{id_}-image.nii.gz")
        label_path = os.path.join(self.label_dir, f"{id_}-label.nii.gz")
        image = nib.load(image_path)
        label = nib.load(label_path)

        image_arr = image.get_fdata().astype(np.float)
        label_arr = label.get_fdata().astype(np.uint8)
        roi_centroids = self.get_roi_centroids(label_arr)

        image_rois = [self.crop_roi(image_arr, centroid) for centroid in roi_centroids]
        label_rois = [self.crop_roi(label_arr, centroid) for centroid in roi_centroids]

        if self.transforms is not None:
            image_rois = [self.apply_transforms(image_roi) for image_roi in image_rois]
        #concatenate in the first dimension, then add a new axis
        image_rois = torch.tensor(np.stack(image_rois)[:, np.newaxis], dtype = torch.float)
        #change 1, 2 in label into 1 by bool operation
        label_rois = (np.stack(label_rois) > 0).astype(np.float)
        label_rois = torch.tensor(label_rois[:, np.newaxis], dtype = torch.float)

        return image_rois, label_rois

    @staticmethod
    def collate_fn(samples):
        image_rois = torch.cat([x[0] for x in samples])
        label_rois = torch.cat([x[1] for x in samples])

        return image_rois, label_rois




class TestDataset(Dataset):

    def __init__(self, image_path, crop_size=64, transforms=None):
        image = nib.load(image_path)
        self.image_affine = image.affine
        self.image = image.get_fdata().astype(np.int16)
        self.crop_size = crop_size
        self.transforms = transforms
        self.centers = self._get_centers()
        self.step = self.crop_size // 2

    def _get_centers(self):
        dim_coords = [list(range(0, dim, self.crop_size // 2))[1:-1]\
            + [dim - self.crop_size // 2] for dim in self.image.shape]
        centers = list(product(*dim_coords))

        return centers

    def __len__(self):
        return len(self.centers)

    def _crop_patch(self, idx):
        center_x, center_y, center_z = self.centers[idx]
        patch = self.image[
            center_x - self.step:center_x + self.step,
            center_y - self.step:center_y + self.step,
            center_z - self.step:center_z + self.step]

        return patch

    def _apply_transforms(self, image):
        for t in self.transforms:
            image = t(image)

        return image

    def __getitem__(self, idx):
        image = self._crop_patch(idx)
        center = self.centers[idx]

        if self.transforms is not None:
            image = self._apply_transforms(image)

        image = torch.tensor(image[np.newaxis], dtype=torch.float)

        return image, center

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x[0] for x in samples])
        centers = [x[1] for x in samples]

        return images, centers

