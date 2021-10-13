# General libraries
import numpy as np
import torch

# torch utils
from torchinfo import summary
from torch.utils.data import random_split

# MDRNN
from mdrnn import MDLSTM

# Image manipulation
from matplotlib.pyplot import imread, imshow, show

# Fila manipulation
from os.path import join as path_join
from os.path import exists as path_exists

# Debug
from tqdm import tqdm
from pdb import set_trace as bp

class AirFreight(torch.utils.data.Dataset):
    """ Ray-traced colour image sequence. Contains 455 frames with 155 distinct
        textures. Frames are 160x120 pixels. """
    def __init__(self, path):
        super().__init__()
        self._path = path
        self._lbl_map = {}
        self.compute_labels()
    def compute_labels(self):
        pbar = tqdm(total=455*120*160)
        for idx in range(455):
            lbl = imread(path_join(self._path, f"afreightseg{idx+1:03}.png"))
            for i in range(120):
                for j in range(160):
                    # Get the label for pixel (i,j) in image idx
                    x = tuple(lbl[i,j])
                    # If not yet seen, map this label to a new integer
                    if x not in self._lbl_map:
                        self._lbl_map[x] = len(self._lbl_map)
                    pbar.update(1)
        pbar.close()
    def reduce_lbl(self, lbl):
        out = np.empty((120,160), dtype=np.int64)
        for i in range(120):
            for j in range(160):
                out[i,j] = self._lbl_map[tuple(lbl[i,j])]
        return out
    def __len__(self):
        return 455
    def __getitem__(self, idx):
        # Load images
        img = imread(path_join(self._path, f"afreightim{idx+1:03}.png")) # The "03" will pad with zeros!
        lbl = imread(path_join(self._path, f"afreightseg{idx+1:03}.png"))
        # Format labels (from a color to an integer representing one of the 157 possible colors).
        lbl = self.reduce_lbl(lbl)
        # Move channel/color dimension at the start
        img = np.moveaxis(img,-1,0)
        return img, lbl

class MyNet(torch.nn.Module):
    def __init__(self, size_in, dim_in):
        super().__init__()
        self.mdrnn1 = MDLSTM(size_in, dim_in, 5)
        self.mdrnn2 = MDLSTM(5, dim_in, 157)
    def forward(self, x):
        # Formatting part
        x = torch.movedim(x, 0, -1) # Put batch_size dimension at the end.
        x = torch.movedim(x, 0, -1) # Put channel info at the end
        # LSTM part
        x = self.mdrnn1(x)          # Multi-dimensional LSTM layer
        x = self.mdrnn2(x)          # Multi-dimensional LSTM layer
        # Formatting part
        x = torch.movedim(x, -1, 0) # Put channel at the start again
        x = torch.movedim(x, -1, 0) # Put batch_size at the start again
        return x

if __name__=="__main__":
    # Load data
    batch_size = 1
    data = AirFreight(path="./data/afreightdata")
    data_train, data_test, data_val = random_split(data, [250, 150, 55])

    # Prepare data loaders
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=2)

    # Define network
    img_shape = data[0][0].shape
    size_in = img_shape[0] # Number of channels for image
    dim_in  = len(img_shape)-1 # Dimensionnality of the "sequence". Here, 2 because it's an image :)
    net = MyNet(size_in, dim_in)
    summary(net)

    # Train network
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-6, momentum=0.9)
    print_step = 5
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass. Squeeze to get final shape.
            outputs = net(inputs)
            # Backward pass
            loss = criterion(outputs, labels)
            loss.backward()
            # Optimization pass
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            if i % print_step == (print_step-1):    # print every [print_step]] mini-batches
                print(f"[{epoch + 1}, {i + 1: 5}] loss: {running_loss/print_step:.3f}")
                running_loss = 0.0
        with torch.no_grad():
            test_loss = 0.
            test_acc = 0.
            for i, data in enumerate(test_loader):
                print(i)
                img, lbl = data
                outputs = net(img)
                # Test loss
                test_loss += criterion(outputs, lbl).item()
                # Test pixel classification accuracy
                _,predicted = torch.max(outputs, dim=1)
                test_acc += (predicted==lbl).sum().item() / (120*160)
            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            print(f"[{epoch+1}] test loss: {test_loss}")
            print(f"[{epoch+1}] test acc: {100*test_acc}%")
    print('Finished Training')

    # Validation step
    with torch.no_grad():
        val_loss = 0.
        val_acc = 0.
        for data in val_loader:
            img, lbl = data
            outputs = net(img)
            val_loss += criterion(outputs, lbl).item()
            _, predicted = torch.max(outputs.data, dim=1)
            test_acc += (predicted==lbl).sum().item() / (120*160)
        val_loss /= len(val_loader)
        val_acc =/ len(val_loader)
        print(f"Validation loss: {val_acc}.")
        print(f"Validation accuracy: {100*test_acc} (pixel classification)/")
