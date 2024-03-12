##########################################################
#Author:          Yufeng Liu
#Create time:     2024-03-06
#Description:               
##########################################################

import numpy as np
import SimpleITK as sitk
from skimage.transform import rescale

from file_io import load_image, save_image
from anatomy.anatomy_config import MASK_CCF25_FILE

__RX_CCF25__, __RY_CCF25__, __RZ_CCF25__ = 216, 18, 228
__ROTATE_Z__ = 5.
__SCALE_Y__ = 0.9434

def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R

def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)

def get_center(img, w, h, d):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    return img.TransformIndexToPhysicalPoint((w,h,d))

def ccf2stereotactic_mask_res25(mask_file, stereo_file=None):
    """
    The majority of this implementation follows the instruction of the resources:
    1. https://community.brain-map.org/t/how-to-transform-ccf-x-y-z-coordinates-into-stereotactic-coordinates/1858
    2. https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-volumetric-data-e-g-mri
    """
    rx, ry, rz = __RX_CCF25__, __RY_CCF25__, __RZ_CCF25__
    theta_z = __ROTATE_Z__
    sx = __SCALE_Y__

    if type(mask_file) is str:
        image = sitk.ReadImage(mask_file)
    elif type(mask_file) is np.ndarray:
        image = sitk.GetImageFromArray(mask_file)
    else:
        raise ValueError('Incorrect input type for argument: `mask_file`')

    theta_z = np.deg2rad(theta_z)
    euler_transform = sitk.Euler3DTransform()
    image_center = get_center(image, rx, ry, rz)    #228,18,216 (z,y,x)
    euler_transform.SetCenter(image_center)

    axis_angle = (0, 0, 1, theta_z)
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_image = resample(image, euler_transform)
    # do scaling
    #scale_transform = sitk.ScaleTransform(3, (1,0.5,1))
    #resampled_image = resample(resampled_image, scale_transform)
    resampled_image = rescale(sitk.GetArrayFromImage(resampled_image), 
                (1,sx,1), order=0, mode='constant', preserve_range=True)
    if stereo_file:
        save_image(stereo_file, resampled_image, useCompression=True)

    return resampled_image

if __name__ == '__main__':
    stereo_file = 'temp.nrrd'
    mask_file = MASK_CCF25_FILE

    rot_img = ccf2stereotactic_mask_res25(mask_file, stereo_file)
    print()




