import torch
import torch.nn as nn

from code.base.base_model import BaseModel
from .sync_layer import SincConv_fast


class FMS(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(sz, sz)

    def forward(self, x):
        r = self.sigmoid(self.fc(torch.mean(x, dim=-1))).unsqueeze(-1)
        return x * r + r


class RB(nn.Module):
    def __init__(self, input_ch, output_ch, hint2=True):
        super().__init__()

        self.lrelu = nn.LeakyReLU(0.3)
        self.hint2 = hint2
        if self.hint2:
            self.bn1 = nn.BatchNorm1d(input_ch)

        self.conv1 = nn.Conv1d(input_ch, output_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(output_ch)
        self.conv2 = nn.Conv1d(output_ch, output_ch, 3, padding=1)
        self.conv1d = nn.Conv1d(input_ch, output_ch, 1)
        self.maxpool = nn.MaxPool1d(3)
        self.fms = FMS(output_ch)
        self.down = (input_ch != output_ch)

    def forward(self, x):
        out = x
        if self.hint2:
            out = self.lrelu(self.bn1(x))

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.maxpool(out + (self.conv1d(x) if self.down else x))
        out = self.fms(out)
        return out


class RawNet2(BaseModel):
    def __init__(self, sync_out_channels, sync_kernel_size, sync_min_low_hz, sync_min_band_hz,
                 rb_sz1, rb_sz2,
                 gru_num_layers, gru_hidden_size):
        super().__init__()

        self.sync_conv = SincConv_fast(sync_out_channels, sync_kernel_size,
                                       min_low_hz=sync_min_low_hz, min_band_hz=sync_min_band_hz)
        self.maxpool = nn.MaxPool1d(3)
        self.bn1 = nn.BatchNorm1d(sync_out_channels)
        self.lrelu = nn.LeakyReLU(0.3)
        self.res_blocks = nn.Sequential(
            RB(sync_out_channels, rb_sz1, hint2=False),
            RB(rb_sz1, rb_sz1),
            RB(rb_sz1, rb_sz2),
            RB(rb_sz2, rb_sz2),
            RB(rb_sz2, rb_sz2),
            RB(rb_sz2, rb_sz2)
        )
        self.bn2 = nn.BatchNorm1d(rb_sz2)
        self.gru = nn.GRU(rb_sz2, gru_hidden_size, gru_num_layers, batch_first=True)
        self.fc = nn.Linear(gru_hidden_size, 1)

    def forward(self, audio, **batch):
        x = self.sync_conv(audio)
        x = self.maxpool(torch.abs(x))
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.res_blocks(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x, _ = self.gru(x.transpose(-1, -2))
        return {'scores': self.fc(x[:, -1]).squeeze(-1)}
