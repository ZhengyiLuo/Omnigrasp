import torch
import numpy as np
import torchvision.transforms as transforms

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, points):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''

        indices = np.random.randint(points.shape[0], size=self.N)
        sampled_points = points[indices, :]

        return sampled_points



def normalize_points_byshapenet(points_n, sub_sample_num=2048):
    point_transform = transforms.Compose([
                SubsamplePointcloud(sub_sample_num)
            ])
    points_n = torch.Tensor(point_transform(points_n))[None, ...]
    shapenet_all_cate_mean = torch.Tensor([[[-0.00229248, -0.02756215,0.1032315]]])
    shapenet_all_cate_std = torch.Tensor([[[0.18059574]]])
    points_n = (points_n.cpu()  - shapenet_all_cate_mean) / shapenet_all_cate_std
    return points_n