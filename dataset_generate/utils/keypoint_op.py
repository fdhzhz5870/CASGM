import collections
import numpy as np
import torch
def filter_points(points, shape, device='cpu'):
    if len(points)!=0:
        mask = (points >= 0) & (points <= torch.tensor(shape, device=device)-1)
        mask = torch.all(mask, dim=1)
        return points[mask]
    else:
        return points
def compute_keypoint_map(points, shape, device='cpu'):
    coord = torch.minimum(torch.round(points).type(torch.int), torch.tensor(shape,device=device)-1)
    kmap = torch.zeros((shape),dtype=torch.int, device=device)
    kmap[coord[:,0],coord[:,1]] = 1
    return kmap
def warp_points(points, homographies, device='cpu'):
    if len(points)==0:
        return points
    points = torch.fliplr(points)
    if len(homographies.shape)==2:
        homographies = homographies.unsqueeze(0)
    B = homographies.shape[0]
    points = torch.cat((points, torch.ones((points.shape[0], 1)).to(points.device)),dim=1)
    warped_points = torch.tensordot(homographies, points.transpose(1,0),dims=([2], [0]))
    warped_points = warped_points.reshape([B, 3, -1])
    warped_points = warped_points.transpose(2, 1)
    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    warped_points = torch.flip(warped_points,dims=(2,))
    warped_points = warped_points.squeeze(dim=0)
    return warped_points
if __name__=='__main__':
    points = torch.tensor([[1.2,0.8],[13,9],[5.6,0.3],[0.4,0.8]])
    homographies = torch.tensor([[[1,0.5,0],[-0.5,1,0],[0,0,1]],[[0.3,0.84,0],[-0.5,1,0],[0,0,1]]])
    pt = warp_points(points, homographies, device='cpu')
    print(pt)
