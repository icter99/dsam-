import torch
import torch.nn as nn

class DepthGeometryPromptHead(nn.Module):
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        self.edge_extract = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, padding=1, groups=16),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, groups=8),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_ch, 1),
        )
        # 初始化为接近零的输出，保证训练初期不干扰原 prompt
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, depth_em):
        geo_prompt = self.edge_extract(depth_em)
        return geo_prompt
