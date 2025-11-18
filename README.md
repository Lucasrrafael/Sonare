# Sonare
projeto inovatech

## Execução

Execute a aplicação principal a partir do diretório do projeto:

```bash
python -m view.main_screen [--debug] [-c CONF] [-d DISPLAY_SECONDS] [-m MODEL]
```

### Parâmetros
- `--debug`: ativa o modo debug (desenha as bounding boxes e rótulos no vídeo)
- `-c, --conf`: threshold de confiança do YOLO (0.0 a 1.0). Padrão: `0.85`
  - Exemplo: `-c 0.7`
- `-d, --display-seconds`: tempo (em segundos) que cada produto permanece na tela. Padrão: `6.0`
  - Exemplo: `-d 8`
- `-t, --carousel-time`: duração (em segundos) de cada imagem no carrossel de backgrounds. Padrão: `3`
  - Exemplo: `-t 5`
- `-m, --model`: caminho para o modelo de detecção (opcional)
  - Se não especificado, o sistema escolhe automaticamente o melhor modelo baseado no hardware

### Suporte a Modelos

A aplicação escolhe **automaticamente** o melhor modelo baseado no hardware disponível:

#### 🚀 Com GPU NVIDIA
- **TensorRT** (`.engine`) - Máxima performance em GPU
- **PyTorch CUDA** (`.pt`) - Alternativa se TensorRT não estiver disponível
- Caminho padrão: `resources/best.engine` ou `resources/best.pt`

#### 💻 Sem GPU (CPU Intel)
- **OpenVINO** - Otimizado para CPUs Intel
- Caminho padrão: `resources/best_openvino_model`
- **PyTorch CPU** (`.pt`) - Fallback se OpenVINO não estiver disponível

#### 🎯 Detecção Automática e Transparente

O sistema detecta automaticamente:
- ✓ Presença de GPU CUDA
- ✓ Modelos disponíveis no sistema
- ✓ Melhor combinação hardware/modelo

**Você não precisa especificar nada** - o sistema escolhe a melhor opção automaticamente!

Exemplo de log ao iniciar:
```
GPU CUDA detectada: NVIDIA GeForce RTX 3080
✓ GPU detectada - usando TensorRT para máxima performance
Modelo TensorRT carregado com sucesso: resources/best.engine
```

ou em CPU:
```
GPU não detectada, usando CPU
✓ CPU detectada - usando OpenVINO para máxima performance em CPU
Modelo OpenVINO carregado com sucesso: resources/best_openvino_model
```

#### Monitoramento de Performance

**Todos os modelos** registram o tempo de inferência detalhado no console em tempo real:

##### Formato de Log
- **OpenVINO**: pré-processamento, inferência, pós-processamento e total
- **TensorRT**: inferência, pós-processamento e total
- **PyTorch**: inferência, pós-processamento e total

##### Exemplos de Log
```
[OpenVINO CPU] Inferência: 25.3ms (pré: 3.1ms, pós: 4.2ms, total: 32.6ms) - 2 detecções
[TensorRT GPU] Inferência: 8.5ms (pós: 1.2ms, total: 9.7ms) - 3 detecções
[YOLO PyTorch] Inferência: 42.8ms (pós: 2.5ms, total: 45.3ms) - 2 detecções
```

##### Interpretação dos Tempos
- **pré**: Conversão de formato, redimensionamento, normalização (apenas OpenVINO)
- **Inferência**: Tempo puro de execução do modelo neural
- **pós**: Extração e formatação das detecções
- **total**: Tempo completo do frame (pré + inferência + pós)

### Exemplos

- Executar com detecção automática (recomendado):
```bash
python -m view.main_screen
```

- Executar com debug e confiança 0.6:
```bash
python -m view.main_screen --debug --conf 0.6
```

- Executar com 8 segundos de exibição por produto:
```bash
python -m view.main_screen -d 8
```

- Forçar uso de modelo PyTorch:
```bash
python -m view.main_screen -m resources/best.pt
```

- Forçar uso de modelo OpenVINO:
```bash
python -m view.main_screen -m resources/best_openvino_model
```

- Forçar uso de modelo TensorRT:
```bash
python -m view.main_screen -m resources/best.engine
```

### Testes e Benchmark

#### Testar Modelos
Para testar se os modelos estão funcionando corretamente:
```bash
python test_models.py
```

#### Benchmark de Performance
Para comparar o desempenho entre os modelos disponíveis:
```bash
python benchmark_models.py
```

Opções do benchmark:
- `-n, --iterations`: Número de iterações (padrão: 10)

Exemplo:
```bash
python benchmark_models.py -n 50
```

O script de benchmark mostrará:
- Tempo médio, mínimo, máximo e mediano
- Desvio padrão
- FPS estimado
- Comparação de velocidade entre os modelos

## Fontes TTF para acentuação no overlay 

> Precisa de revisão

Para exibir acentos nos textos (nome e preço) sobre o vídeo, a aplicação procura primeiro por fontes TrueType no caminho local:

```
resources/fonts/DejaVuSans.ttf
resources/fonts/DejaVuSans-Bold.ttf
```

## Requisitos

### Dependências Python
- Python 3.10+
- Ver todas as dependências em `requirements.txt`

### Hardware Recomendado

#### Para Máxima Performance
- **GPU NVIDIA** com suporte CUDA
- TensorRT instalado
- Modelo: `resources/best.engine`

#### Para CPU Intel
- **OpenVINO** instalado
- Modelo: `resources/best_openvino_model`

#### Fallback Universal
- Qualquer CPU/GPU
- PyTorch instalado
- Modelo: `resources/best.pt`

### Instalação

1. Instalar dependências:
```bash
pip install -r requirements.txt
```

2. (Opcional) Para GPU NVIDIA, instalar TensorRT:
```bash
pip install tensorrt
```

3. Preparar modelos:
   - OpenVINO: colocar em `resources/best_openvino_model/`
   - TensorRT: colocar em `resources/best.engine`
   - PyTorch: colocar em `resources/best.pt`

## Performance Esperada

| Hardware | Modelo | FPS Estimado | Latência |
|----------|--------|--------------|----------|
| RTX 3080 | TensorRT | ~100 fps | ~10 ms |
| RTX 3060 | TensorRT | ~80 fps | ~12 ms |
| Intel i7 | OpenVINO | ~30-40 fps | ~25-35 ms |
| Intel i5 | OpenVINO | ~20-30 fps | ~35-50 ms |
| CPU Genérica | PyTorch | ~10-20 fps | ~50-100 ms |

**Nota**: Valores aproximados, variam conforme configuração específica do sistema.
