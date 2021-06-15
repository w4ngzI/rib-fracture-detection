import nibabel as nib
import numpy as np
from skimage.morphology import binary_dilation, cube, diamond
from tqdm import tqdm

from multiprocessing import Pool

def f(i):
    # print(i, 'starts')
    lung_neighboring_voxel = 5
    aw_neighboring_voxel = 2
    lung_kernel = cube(lung_neighboring_voxel)
    aw_kernel = cube(aw_neighboring_voxel)
    lung_mask = nib.load('./mask/RibFrac' + str(i) + '_lung_mask_25.nii.gz')
    image_affine = lung_mask.affine
    lung_mask = lung_mask.get_fdata().astype(np.int8)
    lung_mask_expand = binary_dilation(lung_mask, lung_kernel).astype(np.int8)
    # aw_mask = nib.load('result/RibFrac' + str(i) + '_aw.nii.gz')
    # aw_mask = aw_mask.get_fdata().astype(np.int8)
    # aw_mask_expand = binary_dilation(aw_mask, aw_kernel).astype(np.int8)
    #
    # nib.Nifti1Image(aw_mask_expand, image_affine).to_filename('./result/RibFrac' + str(i) + '_dilated_aw.nii.gz')
    nib.Nifti1Image(lung_mask_expand, image_affine).to_filename('./mask/RibFrac' + str(i) + '_lung_mask_30.nii.gz')
    print(i, 'ends')

if __name__ == '__main__':

    with Pool(12) as p:
        print(p.map(f, range(421, 501)))


#
# for i in tqdm(range(421, 501)):
#     lung_mask = nib.load('./mask/RibFrac' + str(i) + '_lung_and_aw.nii.gz')
#     image_affine = lung_mask.affine
#     lung_mask = lung_mask.get_fdata().astype(np.int8)
#     lung_mask_expand = binary_dilation(lung_mask, lung_kernel).astype(np.int8)
#     # aw_mask = nib.load('result/RibFrac' + str(i) + '_aw.nii.gz')
#     # aw_mask = aw_mask.get_fdata().astype(np.int8)
#     # aw_mask_expand = binary_dilation(aw_mask, aw_kernel).astype(np.int8)
#     #
#     # nib.Nifti1Image(aw_mask_expand, image_affine).to_filename('./result/RibFrac' + str(i) + '_dilated_aw.nii.gz')
#     nib.Nifti1Image(lung_mask_expand, image_affine).to_filename('./mask/RibFrac' + str(i) + '_lung_mask.nii.gz')
