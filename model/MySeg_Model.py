import torch
from torch import nn
from .Double_Otus import Seg
import torchvision.transforms as transforms
import cv2
from matplotlib import pyplot as plt

class Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self, size):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(size)
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)
        self.up = torch.nn.ConvTranspose2d(channel, channel, 2, 2)

    def forward(self, x):
        out = self.layer(self.up(x))
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out)

        return output


class FeatureMapFusion(nn.Module):
    def __init__(self, channl):
        super(FeatureMapFusion, self).__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(channl),
            nn.ReLU()
        )

    def forward(self, encode_map, GRU_map, Decode_map):
        encoder_con = torch.chunk(encode_map, 2, 1)
        GRU_map_ep = GRU_map.expand(1, encoder_con[1].size(1), encode_map.size(2), encode_map.size(3))
        feature_pro = torch.max(GRU_map_ep, encoder_con[1])
        feature = torch.cat([feature_pro, encoder_con[1]], dim=1)
        # activated_feature_map = self.layer(feature_pro)
        concatenated_feature_map = torch.cat([feature, Decode_map], dim=1)

        return concatenated_feature_map


class MySegNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MySegNet, self).__init__()
        self.c1 = Conv_Block(in_ch, 64)
        self.d1 = DownSample(2)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(2)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(2)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(2)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, out_ch, 3, 1, 1)
        print("num_classes: ", out_ch)

        self.g1 = GRUModel(128, 64, 128)
        self.fu1 = FeatureMapFusion(512)
        self.g2 = GRUModel(128, 64, 128)
        self.fu2 = FeatureMapFusion(256)
        self.g3 = GRUModel(128, 64, 128)
        self.fu3 = FeatureMapFusion(128)
        self.g4 = GRUModel(128, 64, 128)
        self.fu4 = FeatureMapFusion(64)
        self.g5 = GRUModel(128, 64, 128)

        self.fd1 = DownSample(8)
        self.fd2 = DownSample(4)
        self.fd3 = DownSample(2)

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        G1 = self.g1(Seg(x))
        G2 = self.g2(G1)
        G3 = self.g3(G2)
        G4 = self.g4(G3)
        G5 = self.g5(G4)

        O1 = self.c6(self.fu1(R4, self.fd1(G1.unsqueeze(0)), self.u1(R5)))
        O2 = self.c7(self.fu2(R3, self.fd2(G2.unsqueeze(0)), self.u2(O1)))
        O3 = self.c8(self.fu3(R2, self.fd3(G3.unsqueeze(0)), self.u3(O2)))
        O4 = self.c9(self.fu4(R1, G4.unsqueeze(0), self.u4(O3)))

        return self.out(O4)
        # return F.log_softmax(self.out(O4),dim=1)


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.098], std=[0.119]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add a batch dimension
    return image


def weights_init(net, init_type='xavier', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


if __name__ == '__main__':
    x = load_and_preprocess_image('../images/test_1.jpg')
    print(x.shape)
    net = MySegNet(1, 1)
    print("shape: ", net(x).shape)
    print(net)