import bpy
import math
import random
import shutil
import time
from pathlib import Path


def randomly_rotate_object(obj_to_change):
    '''Rotacao aleatoria no eixo Z (yaw) -- preserva a nivelacao do objeto no chao.'''
    obj_to_change.rotation_euler = (0, 0, random.random() * 2 * math.pi)


def randomly_jitter_position(obj_to_change, base_location, xy_range=0.12):
    '''Desloca o objeto em X/Y a partir da posicao-base original (nao da posicao
    do frame anterior), pra variar a oclusao entre mesa e cadeira sem random walk.'''
    obj_to_change.location.x = base_location.x + random.uniform(-xy_range, xy_range)
    obj_to_change.location.y = base_location.y + random.uniform(-xy_range, xy_range)


def randomly_position_camera(camera_container, min_factor=0.0, max_factor=1.0):
    '''A camera segue a curva CameraArcPath (constraint Follow Path) e sempre
    mira no TargetObject (constraint Track To no proprio objeto Camera) --
    so precisa variar o offset_factor ao longo do arco, sem trigonometria.'''
    follow = camera_container.constraints['Follow Path']
    follow.use_fixed_location = True
    follow.offset_factor = random.uniform(min_factor, max_factor)


def randomly_light_scene(light_obj, target, distance=1.8):
    '''Orbita a luz num range de elevacao seguro e varia a energia. Faixa de
    energia calibrada empiricamente pra este arquivo/point light -- valores
    na faixa do arquivo antigo (250-900) estouram a superficie do tampo da
    mesa (grande, quase horizontal) pra branco/pastel em boa parte dos
    angulos de camera do rig novo, independente de roughness/specular do
    material (testado e descartado como causa).'''
    azimuth = random.uniform(0, 2 * math.pi)
    elevation = random.uniform(math.radians(35), math.radians(75))

    light_obj.location = (
        target.x + distance * math.cos(elevation) * math.cos(azimuth),
        target.y + distance * math.cos(elevation) * math.sin(azimuth),
        target.z + distance * math.sin(elevation),
    )
    direction = light_obj.location - target
    light_obj.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
    light_obj.data.energy = random.uniform(25, 60)


def randomly_set_environment(hdri_paths, rotation_range=2 * math.pi):
    '''Troca o fundo/luz-ambiente por um HDRI aleatorio, reaproveitando os
    nodes entre chamadas.'''
    world = bpy.context.scene.world
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    bg_node = nodes.get('Background')

    env_node = nodes.get('RandomEnvTexture')
    if env_node is None:
        env_node = nodes.new(type='ShaderNodeTexEnvironment')
        env_node.name = 'RandomEnvTexture'
        mapping_node = nodes.new(type='ShaderNodeMapping')
        mapping_node.name = 'RandomEnvMapping'
        coord_node = nodes.new(type='ShaderNodeTexCoord')
        coord_node.name = 'RandomEnvCoord'
        links.new(coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], env_node.inputs['Vector'])
        links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    mapping_node = nodes['RandomEnvMapping']

    hdri_path = random.choice(hdri_paths)
    if env_node.image is None or env_node.image.filepath != hdri_path:
        env_node.image = bpy.data.images.load(hdri_path, check_existing=True)

    mapping_node.inputs['Rotation'].default_value = (0, 0, random.uniform(0, rotation_range))
    bg_node.inputs['Strength'].default_value = random.uniform(0.2, 0.4)


def randomly_tint_material(material_to_change, saturation_range, value_range, factor_range):
    '''Reaproveita/insere um Mix Color (RandomTint) entre a textura original
    (Base Color, ja um mapa PBR real -- madeira/plastico com aparencia boa
    por si so) e o Principled BSDF. Os ranges de saturacao/brilho/factor sao
    passados por fora porque mesa e cadeira precisam de calibragem diferente:
    a textura de madeira da mesa ja fica boa "as claras", entao o tint dela
    deve ficar escuro/sutil pra nao lavar o grain pra tom pastel; a cadeira
    aceita mais variacao (inclusive clara) sem parecer errado.'''
    nodes = material_to_change.node_tree.nodes
    links = material_to_change.node_tree.links

    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return

    base_color_input = bsdf.inputs['Base Color']
    mix_node = next((n for n in nodes if n.type == 'MIX' and n.name == 'RandomTint'), None)

    if mix_node is None:
        if not base_color_input.is_linked:
            return
        source_socket = base_color_input.links[0].from_socket
        mix_node = nodes.new(type='ShaderNodeMix')
        mix_node.name = 'RandomTint'
        mix_node.data_type = 'RGBA'
        links.new(source_socket, mix_node.inputs['A'])
        links.new(mix_node.outputs['Result'], base_color_input)

    mix_node.blend_type = 'MIX'

    import colorsys
    hue = random.random()
    saturation = random.uniform(*saturation_range)
    value = random.uniform(*value_range)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    mix_node.inputs['B'].default_value = (r, g, b, 1.0)
    mix_node.inputs['Factor'].default_value = random.uniform(*factor_range)


CHAIR_TINT = dict(saturation_range=(0.2, 0.7), value_range=(0.4, 1.0), factor_range=(0.35, 0.65))
TABLE_TINT = dict(saturation_range=(0.2, 0.5), value_range=(0.1, 0.4), factor_range=(0.2, 0.4))


def randomly_set_composition(chair_objs, table_objs):
    '''25% so cadeira, 25% so mesa, 50% os dois juntos (como sempre foi).
    Fecha o gap de composicao com foto real de produto (objeto sozinho no
    quadro) -- o dataset antes so tinha os dois objetos juntos em 100% dos
    renders, entao o modelo nunca viu um objeto isolado durante o treino,
    mesmo quando cor/luz/angulo batiam com a foto real.'''
    r = random.random()
    if r < 0.25:
        show_chair, show_table = True, False
    elif r < 0.5:
        show_chair, show_table = False, True
    else:
        show_chair, show_table = True, True
    for obj in chair_objs:
        obj.hide_render = not show_chair
    for obj in table_objs:
        obj.hide_render = not show_table
    return show_chair, show_table


def ensure_roughness_floor(material, floor=0.5):
    '''O mapa de roughness real da mesa tem media ~0.32 (bastante glossy).
    Nao e a causa principal do "lavado pra pastel" (isso era a energia da
    luz, ver randomly_light_scene), mas um piso moderado evita brilho de
    plastico/verniz novo demais. Roda uma vez no setup, nao por frame -- e
    correcao de material, nao randomizacao por imagem.'''
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return
    rough_input = bsdf.inputs['Roughness']
    if not rough_input.is_linked:
        return
    if any(n.name == 'RoughnessFloor' for n in nodes):
        return
    source_socket = rough_input.links[0].from_socket
    clamp_node = nodes.new(type='ShaderNodeMath')
    clamp_node.name = 'RoughnessFloor'
    clamp_node.operation = 'MAXIMUM'
    clamp_node.inputs[1].default_value = floor
    links.new(source_socket, clamp_node.inputs[0])
    links.new(clamp_node.outputs['Value'], rough_input)


# ============================================================
# Setup
# ============================================================

scene = bpy.context.scene
# Eevee neste arquivo nao populam o pass "Object Index" que o node ID Mask
# usa (confirmado empiricamente -- mascaras saem vazias com Eevee, certas
# com Cycles) -- por isso Cycles, mesmo sendo mais lento.
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.cycles.samples = 64

hdri_paths = [
    '/Users/maria/Downloads/brown_photostudio_02_8k.hdr',
    '/Users/maria/Downloads/cowboy_town_hall_8k.hdr',
    '/Users/maria/Downloads/glasshouse_interior_8k.hdr',
    '/Users/maria/Downloads/historic_cloister_passage_8k.hdr',
    '/Users/maria/Downloads/relax_inn_seaview_suite_8k.hdr',
]

chair_obj = bpy.data.objects['plastic_monobloc_chair_01']
table_obj = bpy.data.objects['round_wooden_table_01']
table_bolts_obj = bpy.data.objects['round_wooden_table_01_bolts']
camera_container = bpy.data.objects['CameraContainer']
light_obj = bpy.data.objects['Light']
target_loc = bpy.data.objects['TargetObject'].location.copy()

for obj in (chair_obj, table_obj):
    for slot in obj.material_slots:
        if slot.material is not None:
            ensure_roughness_floor(slot.material)

chair_base_loc = chair_obj.location.copy()
table_base_loc = table_obj.location.copy()

output_root = Path('/Users/maria/FastCamp/segmentation_dataset')
scratch_dir = Path('/tmp/blender_seg_scratch')
scratch_dir.mkdir(parents=True, exist_ok=True)

ng = scene.compositing_node_group
out_node = ng.nodes['File Output']
out_node.directory = str(scratch_dir) + '/'

blend_stem = Path(bpy.data.filepath).stem
TABLE_MASK_NAME = f'{blend_stem}Segmentation2'   # ID Mask index 400
CHAIR_MASK_NAME = f'{blend_stem}Segmentation3'   # ID Mask index 6000

splits = [('train', 385), ('val', 110), ('test', 55)]
total_count = sum(n for _, n in splits)

for split_name, _ in splits:
    (output_root / 'images' / split_name).mkdir(parents=True, exist_ok=True)
    (output_root / 'raw_masks' / split_name / 'table').mkdir(parents=True, exist_ok=True)
    (output_root / 'raw_masks' / split_name / 'chair').mkdir(parents=True, exist_ok=True)

start_time = time.time()
image_count = 0

for split_name, n_images in splits:
    print(f'Starting split: {split_name} | {n_images} images')

    for idx in range(n_images):
        randomly_rotate_object(chair_obj)
        randomly_rotate_object(table_obj)
        randomly_jitter_position(chair_obj, chair_base_loc)
        randomly_jitter_position(table_obj, table_base_loc)
        randomly_light_scene(light_obj, target_loc)
        randomly_set_environment(hdri_paths)
        randomly_position_camera(camera_container)
        randomly_set_composition([chair_obj], [table_obj, table_bolts_obj])

        for obj, tint_kwargs in ((chair_obj, CHAIR_TINT), (table_obj, TABLE_TINT)):
            for slot in obj.material_slots:
                if slot.material is not None:
                    randomly_tint_material(slot.material, **tint_kwargs)

        stem = f'{idx:06d}'
        rgb_path = output_root / 'images' / split_name / f'{stem}.png'
        scene.render.filepath = str(rgb_path)

        bpy.ops.render.render(write_still=True)

        shutil.move(str(scratch_dir / f'{TABLE_MASK_NAME}.png'),
                    str(output_root / 'raw_masks' / split_name / 'table' / f'{stem}.png'))
        shutil.move(str(scratch_dir / f'{CHAIR_MASK_NAME}.png'),
                    str(output_root / 'raw_masks' / split_name / 'chair' / f'{stem}.png'))

        image_count += 1
        elapsed = time.time() - start_time
        per_image = elapsed / image_count
        remaining = per_image * (total_count - image_count)
        print(f'{image_count}/{total_count} | {per_image:.1f}s/img | ETA {time.strftime("%M:%S", time.gmtime(remaining))}')

print('DONE')
