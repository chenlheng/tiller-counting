# tiller-counting



```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.datasets as dsets
from torchvision.transforms import transforms
import cv2


class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()
        self.fc = nn.Linear(5, 5)
        self.dp = nn.Dropout(0.5)

    def forward(self, x):

        x = self.fc(x)
        x = self.dp(x)

        return x


if __name__ == '__main__':

    x = torch.FloatTensor([1]*5)
    z = torch.FloatTensor([1]*5)
    print(x)
    net = Net()
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), 0.1)

    optim.zero_grad()
    net.train()
    y = net(x)
    loss = criterion(y, z)
    loss.backward()
    optim.step()
    print(y)
    print(net(x))  # net(x) <> y -> dropout changes every time 

    net.eval()  # only controls the state of some modules, e.g., dropout, but cannot stop backward loss
    optim.zero_grad()
    y = net(x)  # the same as prev net(x) -> with no loss.backward(), the network is fixed
    loss = criterion(y, z)
    loss.backward()
    optim.step()
    print(y)
    print(net(x))

    with torch.no_grad():
        optim.zero_grad()
        y = net(x)
        loss = criterion(y, z)
        # loss.backward() -> torch.no_grad set torch.parameters() to be empty set
        optim.step()
        print(y)
        print(net(x))
```



Output:

```
tensor([ 1.,  1.,  1.,  1.,  1.])
tensor([ 0.0000, -0.0000,  1.6945,  0.7744,  0.0000])
tensor([ 0.0000, -0.0000,  0.0000,  0.9910,  0.0000])
tensor([ 0.3758, -1.1484,  0.5139,  0.4955,  0.7577])
tensor([ 0.5256, -0.6328,  0.6306,  0.6166,  0.8158])
tensor([ 0.5256, -0.6328,  0.6306,  0.6166,  0.8158])
tensor([ 0.5256, -0.6328,  0.6306,  0.6166,  0.8158])
```

