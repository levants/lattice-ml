import torch
from torchvision import transforms


def find_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


class ToTensor(object):

    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def convert(self, x):
        return x if isinstance(x, torch.Tensor) else self.to_tensor(x)

    def __call__(self, x):
        return self.convert(x)


class NetWrapper(object):

    def __init__(self, net, transform, device=None):
        self._net = net.eval()
        self.transform = transform
        self._device = device if device else find_device()
        self._net.to(self._device)
        self.cpu = torch.device('cpu')

    @property
    def net(self):
        return self._net

    @property
    def device(self):
        return self._device

    def __getitem__(self, i):
        return self.net[i]

    def __len__(self):
        return len(self.net)

    @torch.inference_mode()
    def forward(self, *xs, k=6):
        ts = torch.stack(
            [self.transform(x) for x in xs],
            dim=0
        )
        ts = ts.to(self.device)
        rs = self[: k](ts) if k else self.net(ts)
        rs = rs.to(self.cpu).detach().numpy()

        return rs

    def to(self, device=None):
        dvc = device if device else find_device()
        self._device = dvc
        self.net.to(self._device)

    def __call__(self, *xs, k=6):
        return self.forward(*xs, k=k)
