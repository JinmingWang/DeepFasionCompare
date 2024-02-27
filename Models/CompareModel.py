from Models.ModelBasics import *


class CompareModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 先把两张图片分别编码成两个向量
        # 然后把两个向量拼接起来，再通过一个全连接层输出一个值表示两张图片是否是同一个衣服

        self.encoder = nn.Sequential(   # (3, 256, 192)
            ConvBnReLU(3, 32, 3, 2, 1),     # (32, 128, 96)
            ConvBnReLU(32, 64, 3, 2, 1),    # (64, 64, 48)
            ConvBnReLU(64, 128, 3, 2, 1),   # (128, 32, 24)
            ConvBnReLU(128, 256, 3, 2, 1),  # (256, 16, 12)
            ConvBnReLU(256, 512, 3, 2, 1),  # (512, 8, 6)
            ConvBnReLU(512, 1024, 3, 2, 1), # (1024, 4, 3)
            nn.Flatten(),                     # 12288
            nn.Linear(12288, 2048),          # 2048
        )

        self.fc = nn.Sequential(    # 2048
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # img1: (B, 3, 256, 192)
        # img2: (B, 3, 256, 192)

        # img1: (B, 2048)
        # img2: (B, 2048)
        encode_1 = self.encoder(img1)
        encode_2 = self.encoder(img2)

        # x: (B, 1)
        # |A - B| == |B - A|，考虑到了比较两个图像时，不考虑先后顺序，都能得到相同的结果
        x = self.fc(torch.abs(encode_1 - encode_2))

        return x