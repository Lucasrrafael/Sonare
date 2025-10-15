# Sonare
projeto inovatech

## Execução

Execute a aplicação principal a partir do diretório do projeto:

```bash
python -m view.main_screen [--debug] [-c CONF] [-d DISPLAY_SECONDS]
```

### Parâmetros
- `--debug`: ativa o modo debug (desenha as bounding boxes e rótulos no vídeo)
- `-c, --conf`: threshold de confiança do YOLO (0.0 a 1.0). Padrão: `0.85`
  - Exemplo: `-c 0.7`
- `-d, --display-seconds`: tempo (em segundos) que cada produto permanece na tela. Padrão: `6.0`
  - Exemplo: `-d 8`

### Exemplos
- Executar com debug e confiança 0.6:
```bash
python -m view.main_screen --debug --conf 0.6
```

- Executar com 8 segundos de exibição por produto:
```bash
python -m view.main_screen -d 8
```

## Fontes TTF para acentuação no overlay 

> Precisa de revisão

Para exibir acentos nos textos (nome e preço) sobre o vídeo, a aplicação procura primeiro por fontes TrueType no caminho local:

```
resources/fonts/DejaVuSans.ttf
resources/fonts/DejaVuSans-Bold.ttf
```

## Requisitos
- Python 3.10+
- Ver dependências em `requirements.txt`
