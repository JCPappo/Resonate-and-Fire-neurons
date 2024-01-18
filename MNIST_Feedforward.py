from raf import RAF
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import snntorch.functional as SF
from snntorch import spikegen
import matplotlib.pyplot as plt

"""Lets test a simple feedforward network with the resonate and fire neurons on the rate encoded MNIST dataset.
If the first epoch doesn't show any improvement on the test and train accuracy rerun the program to get better
initial weights."""

#Download and transform MNIST dataset
data_path='/tmp/data/mnist_og'
dtype = torch.float

# Select cuda if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

data_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
data_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# Set batch size and create DataLoaders
batch_size = 128
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Define the structure of our network
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10
beta = 0.99
frequency = 30
num_steps = 25 # Number of time steps for rate encoding input

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.raf1 = RAF(frequency, beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.raf2 = RAF(frequency, beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        I1 = self.raf1.init_RAF()
        V1 = self.raf1.init_RAF()
        I2 = self.raf2.init_RAF()
        V2 = self.raf2.init_RAF()

        # Record the final layer
        spk2_rec = []
        V2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            I1, V1, spk1 = self.raf1.forward(cur1, I1, V1)
            cur2 = self.fc2(spk1)
            I2, V2, spk2 = self.raf2.forward(cur2, I2, V2)
            spk2_rec.append(spk2)
            V2_rec.append(V2)

        V2_rec = torch.stack(V2_rec, dim=0)
        
        spk2_rec = torch.stack(spk2_rec, dim=0)
        
        return spk2_rec, V2_rec

# Load the network onto CUDA if available
net = Net().to(device)

# Define functions to get the batch accuracy.
# Pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(num_steps, batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

    return acc*100

def train_printer():
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    acc_train = print_batch_accuracy(spike_data, targets, train=True)
    acc_test = print_batch_accuracy(spike_test_data, test_targets, train=False)
    print("\n")

    return acc_train, acc_test

# Set the loss function and the optimizer
loss = SF.loss.ce_rate_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))

# Set number of epochs
num_epochs = 5

# Lists to append train and test loss values of each batch 
loss_hist = []
test_loss_hist = []

# Lists to append train and test accuracy 
acc_hist = []
test_acc_hist = []
test_total_acc = []

counter = 0

# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        # Rate encode input train data
        spike_data = spikegen.rate(data, num_steps=num_steps) 
        spike_data = spike_data.to(device)
        targets = targets.to(device)

        # Forward pass
        net.train()
        spk_rec, V_rec = net(spike_data.view(num_steps, batch_size, -1))

        # Calculate loss
        loss_val = loss(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            # Rate encode input test data
            spike_test_data = spikegen.rate(test_data, num_steps=num_steps)
            spike_test_data = spike_test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_V = net(spike_test_data.view(num_steps, batch_size, -1))

            #Calculate and store test loss
            test_loss = loss(test_spk, test_targets)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            if counter % 50 == 0:
                acc_train, acc_test = train_printer()
            counter += 1
            iter_counter +=1
            acc_hist.append(acc_train)
            test_acc_hist.append(acc_test)

    # After each epoch we test the model on the whole test dataset
    total = 0
    correct = 0
    
    with torch.no_grad():
      net.eval()
      for data, targets in test_loader:
        spike_data = spikegen.rate(data, num_steps=num_steps)
        spike_data = spike_data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        test_spk, _ = net(spike_data.view(num_steps, batch_size, -1))

        # Calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item() 

    acc_test = 100 * correct / total
    test_total_acc.append(acc_test)
    print(f"Test Set Accuracy: {acc_test:.2f}%")

# Lets plot the results 
fig, ax = plt.subplots((3))
fig.set_size_inches(18.5, 10.5)
fig.tight_layout(pad=4.0)

ax[0].plot(loss_hist)
ax[0].plot(test_loss_hist)
ax[0].set_title("Loss Curves")
ax[0].legend(["Train Loss", "Test Loss"])
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Loss")

ax[1].plot(acc_hist)
ax[1].plot(test_acc_hist)
ax[1].set_title("Accuracy Curves")
ax[1].legend(["Train Accuracy", "Test Accuracy"])
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Accuracy")

ax[2].plot(test_total_acc, color='red')
ax[2].set_title("Total Test Accuracy")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Accuracy")

plt.show()