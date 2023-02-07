import cv2
import torch
from hydra import compose, initialize
from torch import nn
from torchvision import transforms

initialize(version_base=None, config_path="../../config")
cfg = compose(config_name="config")


class CBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBlock, self).__init__()
        assert out_channels % 4 == 0
        self.conv3 = nn.Conv2d(in_channels, out_channels // 4, 3, padding=3 // 2)
        self.conv5 = nn.Conv2d(in_channels, out_channels // 4, 5, padding=5 // 2)
        self.conv7 = nn.Conv2d(in_channels, out_channels // 4, 7, padding=7 // 2)
        self.conv9 = nn.Conv2d(in_channels, out_channels // 4, 9, padding=9 // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(
            self.bn(
                torch.cat(
                    [self.conv3(x), self.conv5(x), self.conv7(x), self.conv9(x)], 1
                )
            )
        )


class NetC(nn.Module):
    def __init__(
        self,
        tag,
        kernel_size=9,
        skip_connections=True,
        batch_norm=True,
        kernel_depth_seed=4,
        network_depth=4,
        act_func=nn.ReLU(),
        initializer=None,
    ):
        super(NetC, self).__init__()
        self.tag = tag
        self.block1 = CBlock(1, 4)
        self.block2 = CBlock(4, 16)
        self.block3 = CBlock(16, 32)
        self.block4 = CBlock(32, 64)
        self.block5 = CBlock(64, 128)
        self.pred = nn.Conv2d(128, 1, 5, padding=5 // 2)

    def forward(self, x):
        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.block4(x)

        x = self.block5(x)

        return self.pred(x)


def initialize_model(model_path):
    model = NetC(tag="encoder")
    model.eval()

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=next(model.parameters()).device,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "cpu" in str(device):
        print(
            "Computation will be very slow! to speed-up computation in the top menu: Runtime->Change runtime type->GPU"
        )

    model.to(device)
    return model


def predict(net, img):
    model_device = next(net.parameters()).device
    sig = nn.Sigmoid()
    toten = transforms.ToTensor()
    norm = transforms.Normalize(mean=[cfg.sheba_mean], std=[cfg.sheba_std])
    if len(img.shape) == 2:
        img = img[..., None]
    img = norm(toten(img).float())[:1][None, ...].float()
    img = img.to(model_device)
    with torch.no_grad():
        pred = net(img)
    sig_pred = sig(pred)

    return sig_pred[0, 0].cpu().numpy()
