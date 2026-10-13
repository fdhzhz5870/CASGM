import torch
import torch.nn.functional as f
def erosion2d(image, strel, origin=(0, 0), border_value=1e6):
    image_pad = f.pad(image, [origin[0], strel.shape[1]-origin[0]-1, origin[1], strel.shape[2]-origin[1]-1], mode='constant', value=border_value)
    image_unfold = f.unfold(image_pad, kernel_size=strel.shape[1])
    strel_flatten = torch.flatten(strel,start_dim=1).unsqueeze(-1)
    diff = image_unfold - strel_flatten
    result, _ = diff.min(dim=1)
    return torch.reshape(result, image.shape)
def pixel_shuffle(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0
    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor
    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor
def pixel_shuffle_inv(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    assert height % scale_factor == 0
    assert width % scale_factor == 0
    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor
    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    tensor = tensor.permute(0, 1, 3, 5, 2, 4)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor
