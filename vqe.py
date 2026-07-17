import models
import torch
import matplotlib.pyplot as plt
import numpy as np
import hamiltonian as hl

n = 4 # number of qudits
dim = 5 # qudit dimension
device = 'cuda' # or cpu if you dont have cuda
epochs = 1000

# Define model and optimizer
model = models.VQE(n=n, dim=dim, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initializes the library for your custom Hamiltonian.
hamiltonian_builder = hl.Hamiltonian(dim=dim, n=n, lambda_param=1.0, device=device)
H = hamiltonian_builder.H
true_ground_energy = hamiltonian_builder.true_ground_energy

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
plt.savefig(f"Resultados{n}_{dim}_{epoch}.png")
plt.show()
