import torch
import sys

print("=== Inicializando Aplicação ===")
print(f"Versão do Python: {sys.version}")
print(f"Versão do PyTorch: {torch.__version__}")

# Detecta dinamicamente o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"-> Dispositivo selecionado pelo PyTorch: {device.type.upper()}")

if device.type == 'cuda':
    print(f"   GPU Detectada: {torch.cuda.get_device_name(0)}")
else:
    print("   Rodando em modo de compatibilidade (CPU apenas).")

# --- Exemplo de tensor rodando no dispositivo selecionado ---
# Isso rodará na CPU na sua máquina e na GPU no Servidor sem mudar uma linha de código!
x = torch.rand(3, 3).to(device)
print(f"\nTensor de teste criado com sucesso no dispositivo: {x.device}")