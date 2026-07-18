import models
import torch
import matplotlib.pyplot as plt
import numpy as np
import hamiltonian as hl

n = 4 # number of qudits
dim = 4 # qudit dimension
device = 'cuda' # or cpu if you dont have cuda
epochs = 1000

# Define model and optimizer
model = models.VQE(n=n, dim=dim, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Initializes the library for your custom Hamiltonian.
hamiltonian_builder = hl.Hamiltonian(dim=dim, n=n, lambda_param=2.0, device=device)
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

# --- EXTRAÇÃO E FILTRAGEM EXCLUSIVA DE ESTADOS FACTÍVEIS ---

with torch.no_grad():
    final_state = model()  # Obtém o estado quântico final
    probabilities = torch.abs(final_state.flatten()) ** 2

probabilities = probabilities.cpu().numpy()
H_dense = H.cpu().numpy()

def index_to_state_str(idx, n, dim):
    state = []
    for _ in range(n):
        state.append(str((idx % dim) + 1))
        idx //= dim
    return "-".join(reversed(state))

# Lista que armazenará APENAS as soluções que atendem às restrições
feasible_solutions = []

for idx in range(dim**n):
    state_str = index_to_state_str(idx, n, dim)
    
    # 1. Verifica IMEDIATAMENTE se o estado é factível (sem repetição de qudits/cidades)
    levels_chosen = state_str.split("-")
    is_feasible = len(levels_chosen) == len(set(levels_chosen))
    
    # 2. Se for factível e tiver sido minimamente explorado pelo circuito, nós guardamos
    if is_feasible and probabilities[idx] > 1e-4:
        state_energy = H_dense[idx, idx].real
        feasible_solutions.append({
            "estado": state_str,
            "energia": state_energy,
            "probabilidade": probabilities[idx]
        })

# 3. Ordena as soluções válidas pela menor energia (menor custo do TSP no topo)
feasible_solutions.sort(key=lambda x: x["energia"])

# --- EXIBIÇÃO DOS RESULTADOS FILTRADOS ---
print("\n" + "="*50)
print("     SOLUÇÕES FACTÍVEIS ENCONTRADAS PELO VQE")
print("="*50)

if len(feasible_solutions) == 0:
    print("❌ Nenhuma solução factível foi encontrada com probabilidade relevante.")
    print("💡 Dica: Tente aumentar o valor de 'lambda_param' no Hamiltoniano")
    print("   para penalizar com mais força as soluções inválidas.")
else:
    print(f"Foram encontradas {len(feasible_solutions)} soluções válidas.\n")
    print(f"{'Ranking':<7} | {'Estado (Rota)':<13} | {'Energia (Custo)':<15} | {'Probabilidade':<13}")
    print("-"*55)
    
    # Exibe todas as soluções factíveis encontradas (ou limita com [:10] se forem muitas)
    for rank, sol in enumerate(feasible_solutions, start=1):
        print(f"#{rank:<5} | {sol['estado']:<13} | {sol['energia']:<15.4f} | {sol['probabilidade']:<13.4%}")

print("="*55)