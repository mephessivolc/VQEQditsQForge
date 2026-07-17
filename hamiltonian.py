import torch

class Hamiltonian:
    def __init__(self, dim, n, lambda_param=1.0, device='cuda'):

        self.dim = dim
        self.device = device
        self.n = n
        self.lambda_param = lambda_param # Ajuste o valor de lambda conforme seu problema
        self.N_1 = dim - 1 # O maior nível do qudit (ex: se dim=3, níveis vão de 0 a 2)
        # Matrizes de distância d_ij (exemplo: matriz aleatória ou definida por você)
        # Ela deve ter tamanho (dim, dim)

        d = torch.randn((dim, dim), device=device, dtype=torch.complex64) 
        self.d = 0.5 * (d + d.conj().T) # Garantindo que as distâncias sejam simétricas/reais

        self._H = self.create_hamiltonian()

    # -----------------------------------------------------------------
    # FUNÇÃO AUXILIAR PARA CRIAR O OPERADOR DE PROJEÇÃO NO ESPAÇO TOTAL
    # -----------------------------------------------------------------
    def get_global_projection(self, qudit_indices, levels):
        """
        Cria um operador projetor no espaço de Hilbert total.
        qudit_indices: lista de índices dos qudits (ex: [t] ou [t, t+1])
        levels: lista dos níveis correspondentes (ex: [i] ou [i, j])
        """
        # Inicializa a lista de operadores com a identidade para cada qudit
        ops = [torch.eye(self.dim, device=self.device, dtype=torch.complex64) for _ in range(self.n)]
        
        # Substitui a identidade pelo projetor |level><level| nos qudits escolhidos
        for qudit_idx, level in zip(qudit_indices, levels):
            P = torch.zeros((self.dim, self.dim), device=self.device, dtype=torch.complex64)
            P[level, level] = 1.0
            ops[qudit_idx] = P
        
        # Faz o produto Kronecker (tensor) de todos os qudits para obter a matriz global
        H_global = ops[0]
        for next_op in ops[1:]:
            H_global = torch.kron(H_global, next_op)
            
        return H_global

    def create_hamiltonian(self):
        # -----------------------------------------------------------------
        # CONSTRUÇÃO DO HAMILTONIANO H_C
        # -----------------------------------------------------------------
        # Inicializa a matriz do Hamiltoniano com zeros
        H = torch.zeros((self.dim**self.n, self.dim**self.n), device=self.device, dtype=torch.complex64)

        # TERMO 1: Soma das distâncias entre instâncias de tempo consecutivas (t e t+1)
        for t in range(self.n - 1): # caminha de t=0 até N-2 (garantindo que t+1 existam)
            for i in range(self.dim):
                for j in range(self.dim):
                    d_ij = self.d[i, j]
                    # Projetor global para |i><i|^{(t)} \otimes |j><j|^{(t+1)}
                    P_ij = self.get_global_projection(qudit_indices=[t, t+1], levels=[i, j])
                    H += d_ij * P_ij

        # TERMO 2: Penalidade lambda para repetição de estados em tempos diferentes (t < s)
        for t in range(self.n):
            for s in range(t + 1, self.n):
                for i in range(self.dim):
                    # Projetor global para |i><i|^{(t)} \otimes |i><i|^{(s)}
                    P_ii = self.get_global_projection(qudit_indices=[t, s], levels=[i, i])
                    H += self.lambda_param * P_ii

        # Garante que o Hamiltoniano final seja estritamente Hermitiano (evita erros numéricos)
        H = 0.5 * (H + H.conj().T)

        return H

    @property
    def true_ground_energy(self):
         # Calcula a energia exata para comparação
        return torch.linalg.eigvalsh(self._H).min().item()

    @property
    def H(self):
        return self._H
    
if __name__ == "__main__":
    # Teste de execução da biblioteca
    h = Hamiltonian(dim=3, n=2, lambda_param=0.5, device="cpu")
    
    print("Matriz Hamiltoniana H obtida com sucesso!")
    print(f"Dimensão da Matriz: {h.H.shape}")
    print(f"Energia exata do estado fundamental: {h.true_ground_energy:.6f}")