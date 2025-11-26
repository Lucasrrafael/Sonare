# Thresholds Customizados por Classe

## Visão Geral

O sistema agora suporta **thresholds de confiança customizados por classe**, que sobrescrevem o threshold básico configurado via linha de comando.

## Como Funciona

### 1. Threshold Básico (Padrão)
- Configurado via argumento `-c` ou `--conf` (padrão: 0.85)
- Exemplo: `python view/main_screen.py -c 0.80`

### 2. Thresholds Customizados por Classe
- Definidos no arquivo `view/camera_screen.py`
- **Sobrescrevem** o threshold básico para classes específicas
- Permitem ajuste fino por produto

## Mapeamento Atual

### Classes e Thresholds (products.json)

| ID | Produto                          | Threshold | Motivo                          |
|----|----------------------------------|-----------|----------------------------------|
| 0  | Coca-Cola 2L                     | 0.81      | Detecção estável                |
| 1  | Fanta Laranja 2L                 | 0.84      | Boa acurácia                    |
| 2  | Feijão Carioca Kicaldo 1kg       | 0.65      | Threshold mais baixo necessário |
| 3  | Feijão Fradinho Kicaldo 1kg      | 0.71      | Menor confiança aceitável       |
| 4  | Feijão Preto Kicaldo 1kg         | 0.83      | Detecção confiável              |
| 5  | Leite Condensado Moça            | 0.82      | Boa performance                 |
| 6  | Leite Italac 1L                  | 0.87      | Alta precisão requerida         |
| 7  | Leite Piracanjuba 1L             | 0.78      | Threshold médio                 |
| 8  | Macarrão Espaguete Massas Paulista 500g | 0.80 | Detecção equilibrada      |
| 9  | Miojo de Tomate                  | -         | **FILTRADO (não detecta)**      |
| 10 | Molho de tomate Pomarola         | 0.86      | Alta confiança                  |
| 11 | Óleo Concordia 900ml             | 0.84      | Boa acurácia                    |

## Como Adicionar/Modificar Thresholds

### Passo 1: Identificar o ID da Classe
Consulte o arquivo `resources/products/products.json` para encontrar o ID da classe.

### Passo 2: Editar o Dicionário
No arquivo `view/camera_screen.py`, localize a variável `self.class_thresholds` e adicione/modifique:

```python
self.class_thresholds = {
    0: 0.81,   # Coca-Cola
    1: 0.84,   # Fanta
    # ... outras classes ...
    12: 0.75,  # Nova classe
}
```

### Passo 3: Testar
Execute o sistema com modo debug para verificar:
```bash
python view/main_screen.py --debug
```

## Logs de Diagnóstico

O sistema exibe logs quando detecções são rejeitadas por threshold:

```
[THRESHOLD] Classe 2 rejeitada: conf=0.620 < threshold=0.650
[THRESHOLD] Classe 6 rejeitada: conf=0.850 < threshold=0.870
```

## Prioridade de Thresholds

1. **Threshold Customizado** (se definido para a classe)
2. **Threshold Básico** (configurado via `-c`)

## Filtragem de Classes

### Classe 9 (Miojo)
A classe 9 está **permanentemente filtrada** e não será detectada, independentemente do threshold.

### Outras Filtragens
- Bounding boxes > 85% da tela são automaticamente filtradas
- Detecções abaixo do threshold customizado são rejeitadas

## Exemplo de Uso

### Detecção Normal
```bash
# Threshold básico 0.85, mas classes usam seus thresholds customizados
python view/main_screen.py -c 0.85
```

### Teste com Threshold Baixo
```bash
# Útil para debug e testar detecções
python view/main_screen.py --debug -c 0.50
```

### Verificar Logs
Os logs mostram:
- `[DETECT]` - Informações gerais de detecção
- `[THRESHOLD]` - Rejeições por threshold
- `[DEBUG]` - Informações do modo debug

## Recomendações

1. **Produtos Difíceis**: Use thresholds mais baixos (0.65-0.75)
2. **Produtos Claros**: Use thresholds mais altos (0.85-0.90)
3. **Teste Sempre**: Use modo debug para validar mudanças
4. **Monitore Logs**: Verifique quantas detecções são rejeitadas

## Ajuste Fino

Para encontrar o threshold ideal para uma classe:

1. Execute com `--debug -c 0.50`
2. Observe os valores de confiança no bbox
3. Defina o threshold ligeiramente abaixo do valor médio observado
4. Teste em diferentes condições de iluminação
5. Ajuste conforme necessário

## Benefícios

- ✅ **Flexibilidade**: Cada classe pode ter seu próprio threshold
- ✅ **Precisão**: Reduz falsos positivos em classes confiáveis
- ✅ **Recall**: Aumenta detecções em classes desafiadoras
- ✅ **Manutenibilidade**: Fácil ajustar sem retreinar o modelo

