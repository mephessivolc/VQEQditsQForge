import quforge.quforge as qf
import torch.nn as nn

class VQE(nn.Module):
    def __init__(self, n=2, dim=3, device='cuda'):
        super().__init__()

        self.n = n
        self.device = device
        
        # Cria a string de estado inicial dinamicamente, ex: '0-0-0' para n=3
        init_state_str = "-".join(["0"] * n)
        self.init = qf.State(init_state_str, dim=dim, device=device)

        self.circuit = qf.Circuit(wires=n, dim=dim, device=device)

        # --- CAMADA VARIACIONAL 1 ---
        # Aplica rotações em todos os pares de transição possíveis para a dimensão escolhida
        for j in range(dim):
            for k in range(j + 1, dim):
                self.circuit.RX(index=[i for i in range(n)], j=j, k=k)
                self.circuit.RY(index=[i for i in range(n)], j=j, k=k)
        
        for j in range(dim - 1):
            self.circuit.RZ(index=[i for i in range(n)], j=j)

        # --- EMARANHAMENTO (Entanglement) ---
        # Aplica CNOTs em cadeia para ligar os qudits vizinhos
        for i in range(n - 1):
            self.circuit.CNOT(index=[i, i + 1])

        # --- CAMADA VARIACIONAL 2 ---
        for j in range(dim):
            for k in range(j + 1, dim):
                self.circuit.RX(index=[i for i in range(n)], j=j, k=k)
                self.circuit.RY(index=[i for i in range(n)], j=j, k=k)
        
        for j in range(dim - 1):
            self.circuit.RZ(index=[i for i in range(n)], j=j)

    def forward(self):
        state = self.circuit(self.init)
        return state