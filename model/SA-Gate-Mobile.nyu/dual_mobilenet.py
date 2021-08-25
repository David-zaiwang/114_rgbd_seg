import functools
import time
import torch
import torch.nn as nn
from net_util import SAGate
import torch.utils.model_zoo as model_zoo

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    it can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor // 2) // divisor * divisor)
    # Make sure that round down does not go down by more that 10 percent
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class BasicMobileNet(nn.Module):
    def __init__(self, width_mult=1, round_nearest=8):
        super(BasicMobileNet, self).__init__()
        block = InvertedResidual
        input_channel = 32

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],  # 0
            [6, 24, 2, 2],  # 1
            [6, 32, 3, 2],  # 2
            [6, 64, 4, 2],  # 3
            [6, 96, 3, 1],  # 4
            [6, 160, 3, 2],  # 5
            [6, 320, 1, 1],  # 6
        ]

        # self.channels = [24, 32, 96, 320]
        self.feat_id = [1, 2, 4, 6]
        self.feat_channel = []

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        # self.feature_0 = ConvBNReLU(3, input_channel, stride=2)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # building inverted residual blocks
        for id, (t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id:
                self.__setattr__("feature_%d" % id, nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = []
        for id in self.feat_id:
            x = self.__getattr__("feature_%d" % id)(x)
            y.append(x)

        return y


def load_model(model, state_dict):
    new_model = model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)


class DualMobileNet(nn.Module):
    def __init__(self, bn_momentum=0.1, pretrain=False):
        super(DualMobileNet, self).__init__()
        self.rgbd_model = BasicMobileNet()
        self.hha_model = BasicMobileNet()

        if pretrain:
            pretrain_model = model_zoo.load_url(model_urls['mobilenet_v2'], progress=True)
            load_model(self.rgbd_model, pretrain_model)
            load_model(self.hha_model, pretrain_model)

        self.sagates = nn.ModuleList([
            SAGate(in_planes=24, out_planes=24, bn_momentum=bn_momentum),
            SAGate(in_planes=32, out_planes=32, bn_momentum=bn_momentum),
            SAGate(in_planes=96, out_planes=96, bn_momentum=bn_momentum),
            SAGate(in_planes=320, out_planes=320, bn_momentum=bn_momentum)
        ])



    def forward(self, x1, x2):

        blocks = []
        merges = []

        feature_rgb_1 = self.rgbd_model.feature_1(x1)
        feature_lla_1 = self.hha_model.feature_1(x2)
        # print("feature_rgb_1.size is {}".format(feature_rgb_1.size()))
        # print("feature_lla_1.size is {}".format(feature_lla_1.size()))

        output, merge = self.sagates[0]([feature_rgb_1, feature_lla_1])
        blocks.append(output)
        merges.append(merge)
        # ---------------------------------

        feature_rgb_2 = self.rgbd_model.feature_2(output[0])
        feature_lla_2 = self.hha_model.feature_2(output[1])

        # print("feature_rgb_2.size is {}".format(feature_rgb_2.size()))
        # print("feature_lla_2.size is {}".format(feature_lla_2.size()))

        output, merge = self.sagates[1]([feature_rgb_2, feature_lla_2])
        blocks.append(output)
        merges.append(merge)
        # ---------------------------------

        feature_rgb_4 = self.rgbd_model.feature_4(output[0])
        feature_lla_4 = self.hha_model.feature_4(output[1])
        #
        # print("feature_rgb_4.size is {}".format(feature_rgb_4.size()))
        # print("feature_lla_4.size is {}".format(feature_lla_4.size()))

        output, merge = self.sagates[2]([feature_rgb_4, feature_lla_4])
        blocks.append(output)
        merges.append(merge)
        # ---------------------------------

        feature_rgb_6 = self.rgbd_model.feature_6(output[0])
        feature_lla_6 = self.hha_model.feature_6(output[1])
        #
        # print("feature_rgb_6.size is {}".format(feature_rgb_6.size()))
        # print("feature_lla_6.size is {}".format(feature_lla_6.size()))

        output, merge = self.sagates[3]([feature_rgb_6, feature_lla_6])
        blocks.append(output)
        merges.append(merge)
        # ---------------------------------

        return blocks, merges




if __name__ == '__main__':

    # mobilenet_model = BasicMobileNet()
    # input = torch.zeros([1, 3, 512, 512])
    # feats = mobilenet_model(input)
    # print(feats[0].size())
    # print(feats[1].size())
    # print(feats[2].size())
    # print(feats[3].size())


    model = DualMobileNet()
    input1 = torch.zeros([1, 3, 224, 224])
    input2 = torch.zeros([1, 3, 224, 224])

    model(input1, input2)

    # input = torch.zeros([1, 3, 512, 512])
    # feats = model(input)
    # print(feats[0].size())
    # print(feats[1].size())
    # print(feats[2].size())
    # print(feats[3].size())
    # print(model)
