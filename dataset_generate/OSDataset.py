import glob
import os
import numpy as np
import cv2
import torch
import torch.utils.data as data
from kornia.geometry.transform import warp_perspective
def get_resized_wh(w, h, resize=None):
    if resize is not None:
        scale = min(resize[0] / h, resize[1] / w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new
def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new
def get_resize_homography(w, h, w_new, h_new):
    resize_homo = np.array([[w_new / w, 0, 0],
                           [0, h_new / h, 0],
                           [0, 0, 1]], dtype=float)
    return resize_homo
def get_padded_image(image, w_new, h_new, resize, ret_mask=False):
    mask = None
    padded = np.zeros(resize, dtype=image.dtype)
    padded[:h_new, :w_new] = image
    if ret_mask:
        mask = np.zeros(resize, dtype=bool)
        mask[:h_new, :w_new] = True
    return padded, mask
def get_resized_img(image, resize=None, df=None):
    h, w = image.shape[:2]
    w_new, h_new = get_resized_wh(w, h, resize=resize)
    w_new, h_new = get_divisible_wh(w_new, h_new, df=df)
    resize_homo = get_resize_homography(w, h, w_new, h_new)
    image = cv2.resize(image, (w_new, h_new))
    resized_img, mask = get_padded_image(image, w_new, h_new, resize, ret_mask=True)
    return resized_img, mask, resize_homo
class OSDataset(data.Dataset):
    def __init__(self, config, mode="test"):
        self.root = config['root']
        self.mode = mode
        self.preprocess = config["preprocess"]
        self.co_flag = config["co_flag"]
        self.img_list = []
        self.Hos_list = []
        self._get_img_list()
    def _get_img_list(self):
        scale_type = ["scale12", "scale24"]
        sname = ["WHU-SEN-City", "SEN1-2"]
        for sc in scale_type:
            for sn in sname:
                opt_path = glob.glob(os.path.normpath(os.path.join(self.root, sc, sn, self.mode, "opt", "*.png")))
                opt_path.sort()
                H_dir = os.path.normpath(os.path.join(self.root, sc, sn, self.mode, 'H'))
                for p in opt_path:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    h_path = os.path.join(H_dir, stem + '.txt')
                    H = np.loadtxt(h_path)
                    self.img_list.append(p)
                    self.Hos_list.append(H)
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        resize = tuple(self.preprocess["resize"])
        opt_path = self.img_list[idx]
        sar_path = self.img_list[idx].replace('opt', 'sar')
        H0_1 = torch.tensor(self.Hos_list[idx], dtype=torch.float).view((3, 3))
        image0 = cv2.imread(opt_path, cv2.IMREAD_GRAYSCALE)
        image1 = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        image0, mask0, resized_homo0 = get_resized_img(image0, resize, self.preprocess["df"])
        image1, mask1, resized_homo1 = get_resized_img(image1, resize, self.preprocess["df"])
        H0_1 = torch.matmul(torch.tensor(resized_homo1, dtype=H0_1.dtype),
                            torch.matmul(H0_1, torch.inverse(torch.tensor(resized_homo0, dtype=H0_1.dtype))))
        image0 = torch.as_tensor(image0 / 255.0, dtype=torch.float)[None]
        image1 = torch.as_tensor(image1 / 255.0, dtype=torch.float)[None]
        mask0 = torch.as_tensor(mask0)
        mask1 = torch.as_tensor(mask1)
        data = {'image0': image0,
                'image1': image1,
                'mask0': mask0,
                'mask1': mask1,
                'H01_gt': H0_1,
                'H10_gt': torch.inverse(H0_1),
                }
        if self.co_flag:
            comask0 = warp_perspective(mask1[None][None].to(torch.float), torch.inverse(H0_1)[None], resize,
                                       align_corners=False)[0][0].to(torch.bool) * mask0
            comask1 = warp_perspective(mask0[None][None].to(torch.float), H0_1[None], resize,
                                       align_corners=False)[0][0].to(torch.bool) * mask1
            data.update({
                'comask0': comask0,
                'comask1': comask1
            })
        if self.mode == "test":
            data.update({
                'source': opt_path
            })
        return data
def warp_and_fuse(imgA, imgB, H, alpha=0.5, tile_size=32):
    h, w = imgB.shape[2:]
    warpedA = warp_perspective(imgA, H, (h, w))
    blended = warpedA * alpha + imgB * (1 - alpha)
    ys = torch.arange(h) // tile_size
    xs = torch.arange(w) // tile_size
    checker = (ys[:, None] + xs[None, :]) % 2
    checker = checker.float().unsqueeze(0).unsqueeze(0)
    checker = checker.expand_as(imgB)
    mosaic = warpedA * checker + imgB * (1 - checker)
    blended = blended.squeeze(dim=1).numpy()
    mosaic = mosaic.squeeze(dim=1).numpy()
    return blended, mosaic
if __name__ == '__main__':
    config = {
        "root": "G:/bishezhengli/data/split_data/",
        "preprocess": {
            "resize": (480, 640),
            "df": 16,
        },
        "co_flag": True,
    }
    dataset = OSDataset(config, mode="test")
    test_loader = data.DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=4, drop_last=True)
    print(len(test_loader))
    save_path = "E:/bishezhengli/data/validate_test/"
    for i, data in enumerate(test_loader):
        blended, mosaic = warp_and_fuse(data["image0"],
                                        data["image1"],
                                        data["H01_gt"],
                                        alpha=0.5,
                                        tile_size=32)
        blended = blended[0] * 255.
        mosaic = mosaic[0] * 255.
        sc, sn, _, _, img_name = data["source"][0].split("\\")[-5:]
        os.makedirs(os.path.join(save_path, sc, sn, "blended"), exist_ok=True)
        os.makedirs(os.path.join(save_path, sc, sn, "mosaic"), exist_ok=True)
        blended_path = os.path.join(save_path, sc, sn, "blended", img_name)
        mosaic_path = os.path.join(save_path, sc, sn, "mosaic", img_name)
        cv2.imwrite(blended_path, blended)
        cv2.imwrite(mosaic_path, mosaic)
