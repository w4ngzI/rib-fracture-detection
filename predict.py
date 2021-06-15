import os
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, remove_small_objects
from tqdm import tqdm
from frac_dataset import TestDataset
import transforms as tsfm
from my_unet import U_Net

device = torch.device('cuda:0')

def remove_low_probs(pred, thresh):
    pred = np.where(pred > thresh, pred, 0)

    return pred


def remove_spine_fp(pred, image, thresh):
    image_bone = image > thresh
    image_bone_2d = image_bone.sum(axis=-1)
    image_bone_2d = ndimage.median_filter(image_bone_2d, 10)
    image_spine = (image_bone_2d > image_bone_2d.max() // 3)
    kernel = disk(7)
    image_spine = ndimage.binary_opening(image_spine, kernel)
    image_spine = ndimage.binary_closing(image_spine, kernel)
    image_spine_label = label(image_spine)
    max_area = 0

    for region in regionprops(image_spine_label):
        if region.area > max_area:
            max_region = region
            max_area = max_region.area
    image_spine = np.zeros_like(image_spine)
    image_spine[
        max_region.bbox[0]:max_region.bbox[2],
        max_region.bbox[1]:max_region.bbox[3]
    ] = max_region.convex_image > 0

    return np.where(image_spine[..., np.newaxis], 0, pred)


def _remove_small_objects(pred, size_thresh):
    pred_bin = pred > 0
    pred_bin = remove_small_objects(pred_bin, size_thresh)
    pred = np.where(pred_bin, pred, 0)

    return pred


def remove_far_from_lung(pred, image_id):
    lung_mask = nib.load('./mask/' + image_id + '_lung_mask_30.nii.gz')
    lung_mask_arr = lung_mask.get_fdata().astype(np.uint8)
    pred_labels = label(pred > 0).astype(np.uint8)
    pred_regions = regionprops(pred_labels)
    for region in pred_regions:
        x = int(region.centroid[0])
        y = int(region.centroid[1])
        z = int(region.centroid[2])
        if lung_mask_arr[x][y][z] == 0:
            for coord in region.coords:
                pred[tuple(coord)] = 0
    return pred


def postprocess(pred, image, prob_thresh, bone_thresh, size_thresh, image_id):
    pred = remove_low_probs(pred, prob_thresh)
    pred = remove_spine_fp(pred, image, bone_thresh)
    pred = _remove_small_objects(pred, size_thresh)
    # remove those far away from the lung
    pred = remove_far_from_lung(pred, image_id)

    return pred

def make_submission_files(pred, image_id, affine):
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]

    pred_label_code = [0] + [1] * int(pred_label.max())
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info


if __name__ == '__main__':
    image_dir =  '/media/l301/D/wz/val_image/'
    pred_dir = '/media/l301/D/wz/pred_result/'
    model_path = '/media/l301/D/wz/bestmodel45.pth'

    prob_thresh = 0.1
    bone_thresh = 300
    size_thresh = 700
    crop_size = 64
    step = crop_size // 2

    batch_size = 16
    num_workers = 4

    #model = U_Net(1, 1, first_out_channels=16)
    model = U_Net()
    model.eval()

    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights['model'])
    #model = nn.DataParallel(model).cuda()
    model = model.to(device)

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]

    image_path_list = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0] for path in image_path_list]

    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        dataset = TestDataset(image_path, transforms=transforms)

        test_dataloader = DataLoader(dataset, batch_size = batch_size, num_workers=num_workers,collate_fn=TestDataset._collate_fn)

        pred = np.zeros(test_dataloader.dataset.image.shape)
    
        with torch.no_grad():
            for _, sample in enumerate(test_dataloader):
                images, centers = sample
                images = images.to(device)
                output = model(images).sigmoid().cpu().numpy().squeeze(axis=1)

                for i in range(len(centers)):
                    center_x, center_y, center_z = centers[i]
                    cur_pred_patch = pred[
                        center_x - step:center_x + step,
                        center_y - step:center_y + step,
                        center_z - step:center_z + step]
                    pred[
                        center_x - step:center_x + step,
                        center_y - step:center_y + step,
                        center_z - step:center_z + step] = np.where(cur_pred_patch > 0, np.mean((output[i],
                        cur_pred_patch), axis=0), output[i])

        pred_result = postprocess(pred, test_dataloader.dataset.image, prob_thresh, bone_thresh, size_thresh, image_id)

        pred_image, pred_info = make_submission_files(pred_result, image_id,dataset.image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(pred_dir, f"{image_id}_pred.nii.gz")
        nib.save(pred_image, pred_path)

        progress.update()

    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(pred_dir, "pred_info.csv"),index=False)

