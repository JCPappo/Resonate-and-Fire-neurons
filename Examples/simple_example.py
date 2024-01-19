from raf import RAF
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""Lets simulate a resonate and fire neuron receiving a simple input of one spike every 10 time steps"""

#Set the decay rate and the frequency of our neuron
beta = 0.99
frequency = 10

#Define the structure of our network
num_inputs = 1 
num_hidden = 1

#Set the number of time steps
num_steps = 1000

#Create lists to store the state of our neuron 
V_rec = []
I_rec = []
spk_rec = []

#Create the periodic input spikes
spk_in = torch.tensor([[1] if ((i%10)==0) else [0] for i in range(num_steps)], dtype=torch.float32)

#Create our simple network with one resonate and fire neuron raf
raf = RAF(frequency, beta)

#Initialize the hidden variables
I = raf.init_RAF()
V = raf.init_RAF()

#Begin the simulation
for step in range(num_steps):
    I, V, spk_out = raf.forward(spk_in[step], I, V)

    I_rec.append(I)
    V_rec.append(V)
    spk_rec.append(spk_out)

#Create a tensor out of every list
spk_rec = torch.stack(spk_rec)
V_rec = torch.stack(V_rec)
I_rec = torch.stack(I_rec)

#Detach from the graph and return a numpy array
spk_rec = spk_rec.detach().numpy()
V_rec = V_rec.detach().numpy()
I_rec = I_rec.detach().numpy()

#Lets plot the results
time_steps = np.arange(0,num_steps,1)
spk_loc = np.array([x for x in range(spk_rec.shape[0]) if spk_rec[x,]==1]) 

fig, ax = plt.subplots((3))
fig.set_size_inches(18.5, 10.5)

ax[0].set_xlim(0,num_steps)
ax[0].set_xticks([])  
ax[0].plot(time_steps, V_rec)
ax[0].set_ylabel('Membrane potential V(t)')

ax[1].set_xlim(0,num_steps)
ax[1].set_xticks([]) 
ax[1].plot(time_steps, I_rec, color='r')
ax[1].set_ylabel('Membrane current I(t)')

ax[2].set_xlim(0,num_steps)
ax[2].set_yticks([]) 
ax[2].vlines(spk_loc, ymin=0, ymax=1, colors='black')
ax[2].set_ylabel('Spikes')

plt.show()