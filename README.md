# Segmentação de Instâncias com Dados Sintéticos — Mesa e Cadeira

Projeto final: geração de um dataset sintético no Blender (renderização +
anotação automática via ID Mask) e treinamento de um modelo de segmentação
de instâncias (YOLOv8-seg) para reconhecer cadeiras plásticas e mesas de
madeira, sem uso de nenhuma foto real no treinamento.

## Problema

Segmentação de instância de dois objetos de mobiliário — cadeira plástica
monobloco e mesa de madeira redonda — a partir de imagens sintéticas geradas
no Blender. As duas classes aparecem juntas em toda imagem, com posição e
rotação independentes, incluindo oclusão parcial de um objeto pelo outro.

## Tecnologias

- **Blender** (Cycles) — geração das imagens e das máscaras de segmentação
  via passe de índice de objeto (Object Index / nó ID Mask no compositor)
- **Python (bpy)** — automação da renderização e randomização de cena
- **OpenCV** — extração de contorno das máscaras e conversão pra polígono
  no formato YOLO-seg
- **Ultralytics YOLOv8-seg** (PyTorch) — treinamento do modelo de
  segmentação de instâncias

## Estrutura do repositório

```
final_project/
├── blender/
│   ├── Atividade10MinhaVersão.blend   # cena com os dois objetos + rig de máscara
│   └── generate_dataset.py             # script de randomização + renderização
├── masks_to_yolo_seg.py                # converte máscaras PNG -> anotação YOLO-seg
├── train.py                            # treino do YOLOv8n-seg
├── sample_dataset/                     # amostra pequena (15 imagens) do dataset gerado
├── exemplos_mascaras/                  # exemplo de RGB + máscaras isoladas por objeto
├── runs/                               # saída do treino: pesos, métricas, gráficos
└── README.md
```

O dataset completo (400 imagens) não está neste repositório por tamanho
(~250MB) — apenas uma amostra de 15 imagens em `sample_dataset/`, suficiente
pra inspecionar o formato. [Link para o dataset completo — adicionar aqui].

## Configuração antes de rodar

Os scripts têm caminhos fixos que apontam pra pastas específicas desta
máquina — ajuste antes de rodar em outro ambiente:

- **`blender/generate_dataset.py`**: a lista `hdri_paths` no topo do arquivo
  aponta pra 5 arquivos `.hdr` (~500MB total, não incluídos no repo por
  tamanho) baixados do [Poly Haven](https://polyhaven.com/hdris) (gratuito):
  `brown_photostudio_02`, `cowboy_town_hall`, `glasshouse_interior`,
  `historic_cloister_passage`, `relax_inn_seaview_suite` (versão 8k). Baixe
  e ajuste os caminhos, ou troque por outros HDRIs de sua escolha. A
  variável `output_root` também precisa apontar pra onde você quer salvar
  o dataset.
- **`masks_to_yolo_seg.py`** e **`train.py`**: a constante `DATA_YAML`/
  `DATASET_ROOT` no topo de cada arquivo precisa apontar pro mesmo
  `output_root` usado acima.

O arquivo `.blend` já vem com as texturas empacotadas internamente (`File >
External Data > Pack Resources`), então não depende de nenhum caminho
externo pra abrir/renderizar corretamente.

## Como reproduzir

### 1. Gerar o dataset no Blender

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background \
  blender/Atividade10MinhaVersão.blend \
  --python blender/generate_dataset.py
```

Gera `images/<split>/NNNNNN.png` (RGB) e `raw_masks/<split>/{table,chair}/NNNNNN.png`
(máscaras binárias por instância).

### 2. Converter máscaras em anotações YOLO-seg

```bash
pip install opencv-python-headless numpy
python masks_to_yolo_seg.py
```

Gera `labels/<split>/NNNNNN.txt` (polígono normalizado por instância) e
`data.yaml`.

### 3. Treinar o modelo

```bash
pip install ultralytics
python train.py
```

Recomendado usar um ambiente virtual separado do passo 2 — `ultralytics`
(PyTorch) e `tensorflow` puxam versões incompatíveis de `numpy` se
instalados juntos.

## Dataset sintético

- **Objetos**: cadeira plástica monobloco (`plastic_monobloc_chair_01`) e
  mesa de madeira redonda (`round_wooden_table_01`), ambas presentes em
  toda imagem, com posição/rotação independentes — dá diversidade real de
  oclusão entre as duas instâncias.
- **Variações**: rotação Z aleatória de cada objeto; posição da câmera
  randomizada em coordenadas esféricas ao redor da cena (azimute 360°,
  elevação 15°-60°, distância 2.2-3.2m); luz principal orbitando (elevação
  35°-75°, energia 400-900W); fundo/iluminação ambiente trocado
  aleatoriamente entre 5 HDRIs; cor dos materiais re-tingida aleatoriamente
  por render.
- **Anotação**: automática, via passe de índice de objeto do Blender
  (`pass_index` único por objeto) + nó ID Mask no compositor — cada máscara
  binária já respeita oclusão real entre os objetos (buffer de
  profundidade), convertida em polígono via `cv2.findContours`.
- **Tamanho**: 400 imagens (280 treino / 80 validação / 40 teste), 640×640.
- **Limitações conhecidas**: todo o dataset é sintético (sem fotos reais no
  treino/validação/teste); os objetos não têm sombra de contato com um
  chão (só com o fundo HDRI), o que os deixa com aparência ligeiramente
  "flutuante" — diferente do dataset de classificação do projeto anterior,
  que tinha um plano de chão pra sombra de contato.

## Resultados

Modelo: YOLOv8n-seg, 60 épocas, imagem 640×640.

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Box | 0.993 | 0.981 | 0.990 | 0.975 |
| Mask | 1.000 | 0.987 | 0.990 | 0.954 |

Por classe (mAP50-95 de máscara): mesa 0.963, cadeira 0.945.

Verificação direta (comparando o conjunto de classes detectado contra o
gabarito, imagem por imagem, no split de validação): 77 de 80 imagens
(96,25%) bateram exatamente. As 3 exceções são casos de oclusão severa (um
objeto quase inteiramente atrás do outro), não confusão entre classes.

**Nota sobre a matriz de confusão**: a matriz de confusão gerada
automaticamente pelo Ultralytics (`runs/.../confusion_matrix_normalized.png`)
mostra só 66-76% de acerto na diagonal, o que contradiz as métricas acima.
Investigamos e confirmamos (via a verificação direta descrita acima) que os
números de mAP são os corretos — a matriz de confusão parece ter um problema
de casamento de caixas quando as duas classes estão sempre próximas/
sobrepostas na mesma imagem. Por isso ela não deve ser citada como está.

Gráficos de treino (`results.png`) e exemplos de predição em imagens de
validação (`val_batch*_pred.jpg`) estão em `runs/segment/runs/furniture_seg/`.
