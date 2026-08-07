# Segmentação de Instâncias com Dados Sintéticos — Mesa e Cadeira

Projeto final: geração de um dataset sintético no Blender (renderização +
anotação automática via ID Mask) e treinamento de um modelo de segmentação
de instâncias (YOLOv8-seg) para reconhecer cadeiras plásticas e mesas de
madeira, sem uso de nenhuma foto real no treinamento.

## Problema

Segmentação de instância de dois objetos, uma cadeira plástica
monobloco e uma mesa de madeira redonda, a partir de imagens geradas no
Blender. Posição e rotação de cada objeto são independentes, incluindo
oclusão parcial de um objeto pelo outro; na maioria das imagens as duas
classes aparecem juntas, mas uma fração é renderizada com só um dos dois
objetos (ver "Dataset sintético" abaixo).

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
│   ├── furniture-instance-segmentation.blend   # cena: objetos + rig de câmera (curva) + máscara
│   └── generate_dataset.py             # script de randomização + renderização
├── masks_to_yolo_seg.py                # converte máscaras PNG -> anotação YOLO-seg
├── train.py                            # treino do YOLOv8n-seg
├── sample_dataset/                     # amostra pequena (15 imagens) do dataset gerado
├── exemplos_mascaras/                  # exemplo de RGB + máscaras isoladas por objeto
├── fotos_reais/                        # fotos reais usadas só pro teste de generalização
├── report_charts/                      # scripts + imagens dos gráficos do relatório
├── runs/                               # saída do treino: pesos, métricas, gráficos
└── README.md
```

O dataset completo (550 imagens) não está neste repositório por tamanho
(~350MB) — apenas uma amostra de 15 imagens em `sample_dataset/`, suficiente
pra inspecionar o formato. 

Link para o dataset completo: [[segmentation-dataset](https://drive.google.com/drive/folders/1LstiqVmybXeKbcBS-D8_eC6wzY_-RoFQ?usp=share_link)].

## Configuração antes de rodar

Os scripts têm caminhos fixos que apontam pra pastas específicas desta
máquina — ajuste antes de rodar em outro ambiente:

- **`blender/generate_dataset.py`**: a lista `hdri_paths` no topo do arquivo
  aponta pra 5 arquivos `.hdr` baixados do [Poly Haven](https://polyhaven.com/hdris):
  `brown_photostudio_02`, `cowboy_town_hall`, `glasshouse_interior`,
  `historic_cloister_passage`, `relax_inn_seaview_suite` (versão 8k). Baixe
  e ajuste os caminhos. A variável `output_root` também precisa apontar pra onde você quer salvar
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
  blender/furniture-instance-segmentation.blend \
  --python blender/generate_dataset.py
```

Motor de render é **Cycles** — o Eevee deste arquivo não popula o passe
"Object Index" que o node ID Mask precisa, então as máscaras sairiam vazias
(ver seção "Generalização pra fotos reais" pra mais detalhes desse e de
outros problemas encontrados/corrigidos).

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

### 4. (Opcional) Testar em fotos reais

Coloque fotos (jpg/png/webp) em `fotos_reais/` e rode:

```bash
python report_charts/make_prediction_examples_reais.py
```

Gera `report_charts/exemplos_predicao_reais.png` com a máscara prevista em
cada foto — útil pra avaliar generalização fora do domínio sintético (ver
seção "Generalização pra fotos reais").

## Dataset sintético

- **Objetos**: cadeira plástica monobloco (`plastic_monobloc_chair_01`) e
  mesa de madeira redonda (`round_wooden_table_01`), com posição/rotação
  independentes — dá diversidade real de oclusão entre as duas instâncias.
- **Variações**:
  - Rotação Z aleatória de cada objeto e jitter de posição em X/Y a partir
    da posição-base (varia a oclusão sem virar random walk)
  - **Câmera**: em vez de coordenadas esféricas calculadas manualmente, um
    rig de curva — a câmera (via `CameraContainer`, constraint Follow Path)
    percorre um arco 3D ao redor da cena, sempre mirando o centro via
    constraint Track To. Um único parâmetro (`offset_factor`, 0 a 1) já dá
    posição e orientação corretas; o arco cobre de quase nível dos olhos
    (elevação ~1°) até visão de cima (~45°)
  - Luz principal (point light) orbitando em elevação 35°-75°, energia
    25-60W (calibrado empiricamente — ver nota abaixo)
  - Fundo/iluminação ambiente trocado aleatoriamente entre 5 HDRIs, com
    força 0.2-0.4
  - **Composição**: 25% dos renders só com a cadeira, 25% só com a mesa,
    50% com os dois juntos (antes era sempre os dois) — fecha (parcialmente)
    o gap de composição com foto real de produto, onde o objeto costuma
    aparecer sozinho no quadro
  - Cor dos materiais re-tingida por render (matiz/saturação/brilho), com
    faixas calibradas por classe — mesa mais escura/sutil (preserva o grão
    da madeira), cadeira com faixa mais ampla (aceita tons claros também)
- **Anotação**: automática, via passe de índice de objeto do Blender
  (`pass_index` único por objeto) + nó ID Mask no compositor — cada máscara
  binária já respeita oclusão real entre os objetos (buffer de
  profundidade), convertida em polígono via `cv2.findContours`.
- **Tamanho**: 550 imagens (385 treino / 110 validação / 55 teste), 640×640.
- **Limitações conhecidas**: todo o dataset é sintético (nenhuma foto real
  entra no treino/validação/teste — fotos reais foram usadas só depois, num
  teste qualitativo à parte, ver seção "Generalização pra fotos reais");
  cadeira e mesa usam sempre a mesma malha 3D (`plastic_monobloc_chair_01` /
  `round_wooden_table_01` — sem diversidade de formato/modelo, só de cor,
  pose e composição); só 5 HDRIs de fundo; os objetos não têm sombra de
  contato com o chão (apenas com o fundo HDRI), o que dá a aparência de que
  estão flutuando.

## Resultados

Modelo: YOLOv8n-seg, 60 épocas, imagem 640×640. Pesos e dataset em
`runs/segment/runs/furniture_seg-4/`.

| | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Box | 0.989 | 0.969 | 0.984 | 0.978 |
| Mask | 0.989 | 0.969 | 0.988 | 0.955 |

Por classe (mAP50-95 de máscara): mesa 0.932, cadeira 0.978.

Verificação direta (comparando o conjunto de classes detectado contra o
gabarito, imagem por imagem, no split de validação): 108 de 110 imagens
(98,2%) bateram exatamente — as 2 exceções são casos de mesa parcialmente
oculta pela cadeira, não confusão entre classes.

**Nota sobre a matriz de confusão**: a matriz de confusão gerada
automaticamente pelo Ultralytics (`runs/.../confusion_matrix_normalized.png`)
mostra só 77-86% de acerto na diagonal, o que contradiz as métricas acima.
Diante dessa contradição foi verificado que os
números de mAP são os corretos — a matriz de confusão parece ter um problema
de casamento de caixas quando as duas classes estão sempre próximas/
sobrepostas na mesma imagem. Por isso ela não deve ser citada como está.

Gráficos estão em `report_charts/`:
convergência treino/validação, comparação por classe, e 4 exemplos de
predição (3 corretos + 1 caso de limitação real encontrado no split de
validação — mesa parcialmente oculta pela cadeira, não detectada).

Os arquivos brutos gerados automaticamente pelo Ultralytics em
`runs/segment/runs/furniture_seg-4/` **não são material de apresentação** —
`train_batch*.jpg` mostra o input já embaralhado pela mosaic augmentation
(4 imagens coladas + recorte/rotação aleatórios: é uma verificação de
pipeline pro desenvolvedor, não é pra ser lido por humanos), e
`val_batch*_pred.jpg`/`val_batch*_labels.jpg` são grades de predição com
caixas grossas e rótulos sobrepostos, difíceis de ler numa apresentação —
por isso foram refeitos como `exemplos_predicao.png`.

## Generalização pra fotos reais

O dataset de treino é 100% sintético. Pra medir o quanto isso importa na
prática, foram coletadas 4 fotos reais de mesa/cadeira (`fotos_reais/`,
não usadas em nenhuma etapa do treino) e rodada a inferência do modelo nelas
— um teste externo, fora do pipeline normal de treino/validação. Resultado
em `report_charts/exemplos_predicao_reais.png`.

**Causas de domain gap identificadas e corrigidas ao longo do projeto**:

- **Cor do material**: a randomização de tint gerava saturação e brilho
  sempre no máximo, variando só o matiz — matematicamente incapaz de gerar
  preto, branco, cinza ou tom dessaturado de madeira real. Corrigido
  randomizando saturação e brilho também, com faixas calibradas por classe
  (mesa mais escura/sutil, cadeira mais ampla). Isso resolveu boa parte dos
  falsos negativos em cadeiras de cor escura.
- **Luz estourando a mesa**: com as texturas PBR reais carregando
  corretamente, a superfície grande e quase horizontal do tampo ficava
  lavada pra tons pastéis sob a energia de luz original (250-900W), mesmo
  com o tint certo por baixo — não era reflexo especular (testado e
  descartado via roughness/specular), era radiância alta demais sendo
  comprimida pelo tone mapping AgX. Corrigido reduzindo a luz pra 25-60W.
- **Composição**: o dataset sempre mostrava os dois objetos juntos, nunca
  um sozinho no quadro — diferente de foto de produto real (objeto
  isolado). Testado adicionar frames com só um objeto (25%/25%/50%): não
  recuperou a detecção em fotos de produto isoladas, então essa hipótese
  foi descartada como causa suficiente, mas manteve-se no dataset final por
  ser uma variação de composição realista de qualquer forma.

**Resultado final**: `Cadeira` generaliza bem em cenas com contexto real
(pátio, grupo, fundo complexo). `Mesa` e fotos de objeto isolado (estilo
produto) continuam com desempenho fraco mesmo depois de todas as correções
acima. Causa mais provável remanescente, não testada por escopo de tempo:
`Mesa` e `Cadeira` usam sempre a mesma malha 3D única — sem diversidade de
formato/design —, e a aparência geral de um render Cycles/HDRI (mesmo bem
calibrado) ainda carrega uma assinatura visual "sintética" que difere de
fotografia real de um jeito que ajuste de parâmetro isolado não fecha.
