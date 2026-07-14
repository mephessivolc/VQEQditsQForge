import quforge.quforge as qf
import torch.nn as nn

class VQE(nn.Module):
    def __init__(self, n=2, dim=3, device='cuda'):
        super().__init__()

        self.n = n
        self.device = device
        self.init = qf.State('0-0', dim=dim, device=device)

        self.circuit = qf.Circuit(wires=n, dim=dim, device=device)

        self.circuit.RX(index=[i for i in range(n)], j=0, k=1)
        self.circuit.RX(index=[i for i in range(n)], j=0, k=2)
        self.circuit.RX(index=[i for i in range(n)], j=1, k=2)
        
        self.circuit.RY(index=[i for i in range(n)], j=0, k=1)
        self.circuit.RY(index=[i for i in range(n)], j=0, k=2)
        self.circuit.RY(index=[i for i in range(n)], j=1, k=2)

        self.circuit.RZ(index=[i for i in range(n)], j=0)
        self.circuit.RZ(index=[i for i in range(n)], j=1)

        self.circuit.CNOT(index=[0,1])

        self.circuit.RX(index=[i for i in range(n)], j=0, k=1)
        self.circuit.RX(index=[i for i in range(n)], j=0, k=2)
        self.circuit.RX(index=[i for i in range(n)], j=1, k=2)
        
        self.circuit.RY(index=[i for i in range(n)], j=0, k=1)
        self.circuit.RY(index=[i for i in range(n)], j=0, k=2)
        self.circuit.RY(index=[i for i in range(n)], j=1, k=2)

        self.circuit.RZ(index=[i for i in range(n)], j=0)
        self.circuit.RZ(index=[i for i in range(n)], j=1)

    def forward(self):

        state = self.circuit(self.init)

        return state