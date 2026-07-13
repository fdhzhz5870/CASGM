import cv2
import os
from math import pi
import torch.nn.functional as F
from numpy.random import uniform
from scipy.stats import truncnorm
import kornia
from kornia.geometry.transform import warp_perspective
from .params import dict_update
from .tensor_op import erosion2d
from .keypoint_op import *
from imgaug import augmenters as iaa
def homographic_aug_pipline(img, pts, config, H, device='cpu'):
    if len(img.shape)==2:
        img = img.unsqueeze(dim=0).unsqueeze(dim=0)
    image_shape = img.shape[2:]
    homography = H
    warped_image = warp_perspective(img, homography, image_shape, align_corners=True)
    warped_valid_mask = compute_valid_mask(image_shape, homography, config['valid_border_margin'], device=device)
    warped_points = warp_points(pts, homography, device=device)
    warped_points = filter_points(warped_points, image_shape, device=device)
    warped_points_map = compute_keypoint_map(warped_points, img.shape[2:], device=device)
    return {'warp':{'img': warped_image.squeeze(),
                    'kpts': warped_points,
                    'kpts_map': warped_points_map.squeeze(),
                    'mask':warped_valid_mask.squeeze()}}
def compute_valid_mask(image_shape, homographies, erosion_radius=0, device='cpu'):
    if len(homographies.shape)==2:
        homographies = homographies.unsqueeze(0)
    B = homographies.shape[0]
    img_one = torch.ones(tuple([B,1,*image_shape]),device=device, dtype=torch.float32)
    mask = warp_perspective(img_one, homographies, tuple(image_shape), align_corners=False)
    mask = mask.round()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        kernel = torch.as_tensor(kernel[np.newaxis,:,:],device=device)
        _, kH, kW = kernel.shape
        origin = ((kH-1)//2, (kW-1)//2)
        mask = erosion2d(mask, torch.flip(kernel, dims=[1,2]), origin=origin) + 1.
    return mask.squeeze(dim=1)
def sample_homography(shape, config=None, device='cpu'):
    default_config = {'perspective':True, 'scaling':True, 'rotation':True, 'translation':True,
    'n_scales':10, 'n_angles':50, 'scaling_amplitude':0.2, 'perspective_amplitude_x':0.1,
    'perspective_amplitude_y':0.1, 'patch_ratio':0.5, 'max_angle':pi / 2,
    'allow_artifacts': False, 'translation_overflow': 0., 'std_trunc': 4, 'stype': 0}
    if config is not None:
        config = dict_update(default_config, config)
    else:
        config = default_config
    std_trunc = config['std_trunc']
    margin = (1 - config['patch_ratio']) / 2
    pts1 = margin + np.array([[0, 0],
                              [0, config['patch_ratio']],
                              [config['patch_ratio'], config['patch_ratio']],
                              [config['patch_ratio'], 0]])
    pts2 = pts1.copy()
    if config['perspective']:
        if not config['allow_artifacts']:
            perspective_amplitude_x = min(config['perspective_amplitude_x'], margin)
            perspective_amplitude_y = min(config['perspective_amplitude_y'], margin)
        else:
            perspective_amplitude_x = config['perspective_amplitude_x']
            perspective_amplitude_y = config['perspective_amplitude_y']
        perspective_displacement = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_y/2).rvs(1)
        h_displacement_left = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
        h_displacement_right = truncnorm(-std_trunc, std_trunc, loc=0., scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()
    if config['scaling']:
        if config['stype'] == 0:
            a, b = 0.5, 1
            std = 0.1
            loc = (a + b) / 2
            scale = std / 0.05
            a_trunc = (a - loc) / scale
            b_trunc = (b - loc) / scale
            scales = truncnorm(a_trunc, b_trunc, loc=loc, scale=scale).rvs(config['n_scales'])
        elif config['stype'] == 1:
            a, b = 0.25, 0.5
            std = 0.1
            loc = (a + b) / 2
            scale = std / 0.05
            a_trunc = (a - loc) / scale
            b_trunc = (b - loc) / scale
            scales = truncnorm(a_trunc, b_trunc, loc=loc, scale=scale).rvs(config['n_scales'])
        elif config['stype'] == 2:
            a, b = 2.0, 4.0
            std = 0.1
            loc = (a + b) / 2
            scale = std / 0.05
            a_trunc = (a - loc) / scale
            b_trunc = (b - loc) / scale
            scales = truncnorm(a_trunc, b_trunc, loc=loc, scale=scale).rvs(config['n_scales'])
        elif config['stype'] == 3:
            a, b = 0.25, 1
            std = 0.1
            loc = (a + b) / 2
            scale = std / 0.05
            a_trunc = (a - loc) / scale
            b_trunc = (b - loc) / scale
            scales = truncnorm(a_trunc, b_trunc, loc=loc, scale=scale).rvs(config['n_scales'])
        elif config['stype'] == 4:
            n_scales_half = config['n_scales'] // 2
            a_large, b_large = 4.0, 6.0
            std_large = 0.3
            loc_large = (a_large + b_large) / 2
            scale_large = std_large / 0.05
            a_trunc_large = (a_large - loc_large) / scale_large
            b_trunc_large = (b_large - loc_large) / scale_large
            scales_large = truncnorm(a_trunc_large, b_trunc_large, loc=loc_large, scale=scale_large).rvs(n_scales_half)
            a_small, b_small = 1.0/6.0, 1.0/4.0
            std_small = 0.015
            loc_small = (a_small + b_small) / 2
            scale_small = std_small / 0.05
            a_trunc_small = (a_small - loc_small) / scale_small
            b_trunc_small = (b_small - loc_small) / scale_small
            scales_small = truncnorm(a_trunc_small, b_trunc_small, loc=loc_small, scale=scale_small).rvs(config['n_scales'] - n_scales_half)
            scales = np.concatenate([scales_large, scales_small])
        else:
            a, b = 0.25, 0.5
            std = 0.1
            loc = (a + b) / 2
            scale = std / 0.05
            a_trunc = (a - loc) / scale
            b_trunc = (b - loc) / scale
            scales = truncnorm(a_trunc, b_trunc, loc=loc, scale=scale).rvs(config['n_scales'])
        if config.get('stype', 0) not in [1, 2, 4]:
            scales = np.concatenate((np.array([1]), scales), axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if config['allow_artifacts']:
            valid = np.arange(config['n_scales'])
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]
    if config['translation']:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if config['allow_artifacts']:
            t_min += config['translation_overflow']
            t_max += config['translation_overflow']
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T
    if config['rotation']:
        angles = np.linspace(-config['max_angle'], config['max_angle'], num=config['n_angles'])
        angles = np.concatenate((np.array([0.]),angles), axis=0)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul((pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if config['allow_artifacts']:
            valid = np.arange(config['n_angles'])
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]
    shape = np.array(shape[::-1])
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]
    homography = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))
    homography = torch.tensor(homography,device=device, dtype=torch.float32).unsqueeze(dim=0)
    homography = torch.inverse(homography)
    return homography
def ratio_preserving_resize(img, target_size):
    scales = np.array((target_size[0]/img.shape[0], target_size[1]/img.shape[1]))
    new_size = np.round(np.array(img.shape)*np.max(scales)).astype(np.int)
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_h, target_w = target_size
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp),keep_size=False),])
    new_img = aug(images=temp_img)
    return new_img
