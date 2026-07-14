import models
import torch
import matplotlib.pyplot as plt
import numpy as np

n = 2 # number of qudits
dim = 3 # qudit dimension
device = 'cuda' # or cpu if you dont have cuda
epochs = 1000

# Define model and optimizer
model = models.VQE(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define Hamiltonian, here in this example we use a random hamiltonian
H = torch.randn((dim**n, dim**n), device=device, dtype=torch.complex64)
H = 0.5 * (H + H.conj().T)

true_ground_energy = torch.linalg.eigvalsh(H).min().item()

# Train the model
energies = []
for epoch in range(epochs):

    output_state = model()

    energy = (output_state.conj().T @ H @ output_state).real

    loss = energy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())
    energies.append(energy.item())

print('True ground energy = %.6f' % true_ground_energy)
print('Lowest energy found = %.6f' % np.min(energies))

plt.plot(energies)
plt.axhline(y=true_ground_energy, color='r', linestyle='--')
plt.grid()
plt.show()


