import torch
from torchvision import datasets
from torchvision import transforms

from mdrnn import MDLSTM

from pdb import set_trace as bp

class MyNet(torch.nn.Module):
    def __init__(self, size_in, dim_in, hidden_size):
        super().__init__()
        self.mdrnn1  = MDLSTM(size_in, dim_in, hidden_size)
        self.flatten = torch.nn.Flatten()
        self.lin1    = torch.nn.Linear(hidden_size*28*28,10)
    def forward(self, x):
        # LSTM part
        x = torch.movedim(x, 0, -1) # Put batch_size dimension at the end.
        x = torch.movedim(x, 0, -1) # Put channel info at the end
        x = self.mdrnn1(x)          # Multi-dimensional LSTM layer
        x = torch.movedim(x, -2, 0) # Put batch_size at the start again
        # Linear part
        x = self.flatten(x) # Flatten layer
        x = self.lin1(x)    # Linear layer
        return x

if __name__=="__main__":
    # Load data
    batch_size = 64
    train_set = datasets.MNIST(
        root="data", train=True,
        download=True,
        transform=transforms.ToTensor()
        )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_set = datasets.MNIST(
        root="data", train=False,
        download=True,
        transform=transforms.ToTensor()
        )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Define network
    img_shape = train_set[0][0].shape
    size_in = img_shape[0] # Number of channels for image
    dim_in  = len(img_shape)-1 # Dimensionnality of the "sequence". Here, 2 because it's an image :)
    hidden_size = 5
    net = MyNet(size_in, dim_in, hidden_size)

    # Train network
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print_step = 10
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % print_step == (print_step-1):    # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1: 5}] loss: {running_loss/print_step:.3f}")
                running_loss = 0.0
    print('Finished Training')

    # Test network
    correct = 0
    total = 0
    with torch.no_grad(): # No need for gradients when testing
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Accuracy of the network on the 10000 test images: {(100 * correct / total)}%.")
