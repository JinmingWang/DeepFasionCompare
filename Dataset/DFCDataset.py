import torch
from torch.utils.data import Dataset
from typing import *
import os


class DFCDataset(Dataset):
    def __init__(self, pth_folder: str):
        # 初始化数据集
        # 需要将数据集的路径等信息，如果内存足够，那么就把整个数据集直接加载到内存中
        # 如果内存不够，那么就要存储每个数据的路径，然后在__getitem__中读取数据

        # 文件名格式：
        # 0000026501_0.pth
        # 0000026501_1.pth
        # 0000026501_2.pth
        # 0000008404_0.pth
        # 0000008404_1.pth
        # 0000008404_2.pth
        # ...
        # 10位id表示衣服_该衣服的第几张图片.pth

        self.img_paths: List[str] = []     # 长度为N的列表，每个元素是一个图片的路径
        self.cloth_ids: List[str] = []     # 长度为N的列表，每个元素是一个图片的衣服id

        for file_name in os.listdir(pth_folder):
            self.img_paths.append(os.path.join(pth_folder, file_name))
            self.cloth_ids.append(file_name.split("_")[0])

        self.N = len(self.img_paths)        # 数据集中图片的数量

    def __len__(self) -> int:
        # 因为要做图像比较算法，所以一次要加载2张图片
        # 一共有N张图片，就有N*N对图片
        # 获取数据集长度，也就是一共有多少对图片
        return self.N * self.N

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 获取数据集中的一对图片
        # idx是一个整数，表示第idx对图片
        # 第r行第c列的图片，就是第r*N+c对图片 (行列 -> 绝对索引)
        # (绝对索引 -> 行列) r = idx // N, c = idx % N
        # 就可以把idx一个整数拆分成r和c两个整数，表示第r张和第c张图片
        r: int = idx // self.N
        c: int = idx % self.N

        # 加载每个图已经被存成了torch.Tensor, float32, (C, H, W)
        img_path_1 = self.img_paths[r]
        img_path_2 = self.img_paths[c]

        img_1 = torch.load(img_path_1).cuda()    # (3, H, W)
        img_2 = torch.load(img_path_2).cuda()    # (3, H, W)

        # 尺寸: (1)
        # 两张图片是否是同一个衣服
        label = torch.tensor([self.cloth_ids[r] == self.cloth_ids[c]], dtype=torch.float32, device="cuda")

        return img_1, img_2, label


def collateFunc(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 当你使用pytorch提供的DataLoader加载数据集的时候，需要指定一个collate函数
    # DataLoader会收集多个__getitem__返回的结果，保存成一个list，形如[(__getitem__返回的结果), (__getitem__返回的结果), ...]

    # batch是一个列表，里面的每个元素都是__getitem__返回的结果
    # 每个元素都是(img_1, img_2, label)
    # 需要把这些数据整理成一个batch，返回
    img_1_list, img_2_list, label_list = zip(*batch)    # 分别把img_1, img_2, label分离出来

    # 把img_1_list, img_2_list, label_list转换成一个batch
    batch_img_1 = torch.stack(img_1_list, dim=0)    # (B, 3, H, W)
    batch_img_2 = torch.stack(img_2_list, dim=0)    # (B, 3, H, W)
    batch_label = torch.cat(label_list, dim=0)      # (B)

    return batch_img_1, batch_img_2, batch_label