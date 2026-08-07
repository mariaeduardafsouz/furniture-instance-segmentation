import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
SLOT1_BLUE = '#2a78d6'
SLOT2_ORANGE = '#eb6834'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica Neue', 'Arial', 'DejaVu Sans']

CSV_PATH = '/Users/maria/furniture-instance-segmentation/runs/segment/runs/furniture_seg-4/results.csv'

epochs, train_seg, val_seg = [], [], []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        epochs.append(int(row['epoch']))
        train_seg.append(float(row['train/seg_loss']))
        val_seg.append(float(row['val/seg_loss']))

# ============================================================
# Chart A — convergencia (loss de mascara, treino vs validacao)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(top=0.80, bottom=0.14, left=0.07, right=0.98)

ax.plot(epochs, train_seg, color=SLOT1_BLUE, linewidth=2, solid_capstyle='round', zorder=3)
ax.plot(epochs, val_seg, color=SLOT2_ORANGE, linewidth=2, solid_capstyle='round', zorder=3)

# marcador + rotulo direto no fim de cada linha
for y, color, label in [(train_seg[-1], SLOT1_BLUE, 'Treino'), (val_seg[-1], SLOT2_ORANGE, 'Validação')]:
    ax.scatter([epochs[-1]], [y], s=64, color=color, zorder=4, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f'{label}  {y:.2f}', xy=(epochs[-1], y), xytext=(8, 0),
                textcoords='offset points', va='center', ha='left',
                fontsize=11, color=INK_PRIMARY, fontweight='medium')

ax.set_xlim(epochs[0], epochs[-1] + 9)
ax.set_ylim(0, max(train_seg[0], val_seg[0]) * 1.08)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color(BASELINE)
ax.spines['bottom'].set_linewidth(1)

ax.yaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis='y', length=0, labelsize=10, colors=INK_MUTED)
ax.tick_params(axis='x', length=0, labelsize=10, colors=INK_MUTED)
ax.set_xlabel('Época', fontsize=10, color=INK_SECONDARY)

fig.text(0.07, 0.94, 'Perda de máscara cai de forma consistente em treino e validação',
          fontsize=14, color=INK_PRIMARY, fontweight='bold', ha='left', va='top')
fig.text(0.07, 0.87, 'Gap aumenta nas últimas 10 épocas (mosaic augmentation desligado — padrão do Ultralytics)',
          fontsize=10.5, color=INK_SECONDARY, ha='left', va='top')

plt.savefig('/Users/maria/furniture-instance-segmentation/report_charts/convergencia_treino.png', facecolor=SURFACE, bbox_inches='tight', pad_inches=0.25)
plt.close()

# ============================================================
# Chart B — metricas finais por classe (mascara)
# ============================================================
metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
mesa =    [0.983, 0.938, 0.980, 0.932]
cadeira = [0.996, 1.000, 0.995, 0.978]

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

x = range(len(metrics))
bar_w = 0.32
gap = 0.02

bars1 = ax.bar([i - bar_w/2 - gap/2 for i in x], mesa, width=bar_w,
               color=SLOT1_BLUE, zorder=3)
bars2 = ax.bar([i + bar_w/2 + gap/2 for i in x], cadeira, width=bar_w,
               color=SLOT2_ORANGE, zorder=3)

for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, color=INK_PRIMARY)

ax.set_xticks(list(x))
ax.set_xticklabels(metrics, fontsize=11, color=INK_SECONDARY)
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0', '0.25', '0.50', '0.75', '1.00'], fontsize=10, color=INK_MUTED)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.yaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(length=0)

legend = ax.legend(
    [bars1, bars2], ['Mesa de madeira', 'Cadeira plástica'],
    loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False,
    fontsize=11, labelcolor=INK_PRIMARY, handlelength=1.2, handleheight=1.2,
    columnspacing=1.5,
)

ax.set_title('As duas classes têm desempenho equivalente',
              fontsize=14, color=INK_PRIMARY, fontweight='bold', loc='left', pad=44)

plt.tight_layout()
plt.savefig('/Users/maria/furniture-instance-segmentation/report_charts/metricas_por_classe.png', facecolor=SURFACE, bbox_inches='tight')
plt.close()

print('OK: report_charts/convergencia_treino.png, report_charts/metricas_por_classe.png')
