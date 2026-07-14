# Usamos uma imagem oficial do PyTorch com CUDA 12.1.
# Ela é otimizada e garante paridade total entre sua máquina e o servidor.
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Define o diretório de trabalho
WORKDIR /app

# Instala dependências do sistema que sua biblioteca possa precisar (opcional)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as suas dependências do Python (onde estará a sua biblioteca baseada em PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código
COPY . .

# Comando para rodar a aplicação
CMD ["python", "main.py"]