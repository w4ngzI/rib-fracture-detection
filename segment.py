import numpy as np
import nibabel as nib
from utils import *
from segment_lung import segment_lung
from segment_airway import segment_airway
from tqdm import tqdm
from skimage.morphology import binary_dilation, cube
from multiprocessing import Pool

params = define_parameter()


def f(i):
    # print(i, 'starts')
    lung_neighboring_voxel = 5
    lung_kernel = cube(lung_neighboring_voxel)
    I = nib.load("/Users/xuezhengrong/Desktop/ribfrac-test-images/RibFrac" + str(i) + "-image.nii.gz")
    I_affine = I.affine
    I = I.get_fdata()

    lung_mask = segment_lung(params, I, I_affine)
    nib.Nifti1Image(lung_mask, I_affine).to_filename('./test_mask/RibFrac' + str(i) + '_lung_and_aw.nii.gz')

    for j in range(1, 7):
        lung_mask = binary_dilation(lung_mask, lung_kernel).astype(np.int8)
    # aw_mask = nib.load('result/RibFrac' + str(i) + '_aw.nii.gz')
    # aw_mask = aw_mask.get_fdata().astype(np.int8)
    # aw_mask_expand = binary_dilation(aw_mask, aw_kernel).astype(np.int8)
    #
    # nib.Nifti1Image(aw_mask_expand, image_affine).to_filename('./result/RibFrac' + str(i) + '_dilated_aw.nii.gz')
    nib.Nifti1Image(lung_mask, I_affine).to_filename('./test_mask/RibFrac' + str(i) + '_lung_mask_30.nii.gz')
    print(i, 'ends')

if __name__ == '__main__':

    with Pool(1) as p:
        print(p.map(f, range(653, 654)))

