import bpy
import math
import random
import shutil
import time
from mathutils import Euler, Vector
from pathlib import Path

def randomly_rotate_object(obj_to_change):
    '''Rotação aleatória só no eixo Z (yaw), mesmo padrão do Atividade9 —
    preserva a nivelação original do objeto no chão.'''
    random_rot = (0, 0, random.random() * 2 * math.pi)
    obj_to_change.rotation_euler = Euler(random_rot, 'XYZ')


def randomly_jitter_position(obj_to_change, base_location, xy_range=0.12):
    '''Desloca o objeto em X/Y a partir da posição-base original dele na
    cena (não a partir da posição do frame anterior — mesmo cuidado do
    Atividade9 pra não virar random walk). Isso varia o quanto cadeira e
    mesa se sobrepõem/afastam entre uma imagem e outra, dando diversidade
    de oclusão real pra segmentação de instâncias — sem isso as duas
    ficariam sempre na mesma posição relativa em todo frame.'''
    dx = random.uniform(-xy_range, xy_range)
    dy = random.uniform(-xy_range, xy_range)
    obj_to_change.location.x = base_location.x + dx
    obj_to_change.location.y = base_location.y + dy


def randomly_light_scene(light_name='Light', target=(0.4, -0.17, -0.1), distance=1.8):
    '''Mesmo padrão do Atividade9: órbita a luz num range de elevação seguro
    e varia a energia. Point light aqui (não Area), e a distância/energia
    foram recalibradas pra escala desta cena (objetos ~1m, não ~0.1m).'''
    light_obj = bpy.context.scene.objects.get(light_name)
    if light_obj is None:
        print(f'Aviso: objeto de luz "{light_name}" não encontrado — pulando')
        return

    azimuth = random.uniform(0, 2 * math.pi)
    elevation = random.uniform(math.radians(35), math.radians(75))

    x = target[0] + distance * math.cos(elevation) * math.cos(azimuth)
    y = target[1] + distance * math.cos(elevation) * math.sin(azimuth)
    z = target[2] + distance * math.sin(elevation)
    light_obj.location = (x, y, z)

    direction = light_obj.location - Vector(target)
    light_obj.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()

    if light_obj.data.type == 'SUN':
        light_obj.data.energy = random.uniform(2.5, 4.5)
    elif light_obj.data.type == 'POINT':
        light_obj.data.energy = random.uniform(400, 900)
    else:
        light_obj.data.energy = random.uniform(600, 1200)


def randomly_set_environment(hdri_paths, rotation_range=2 * math.pi):
    '''Idêntica à versão do Atividade9 — troca o fundo/luz-ambiente plano
    por um HDRI aleatório, reaproveitando os nodes entre chamadas.'''
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        print('Aviso: World sem nodes — pulando randomização de ambiente')
        return

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    bg_node = nodes.get('Background')
    if bg_node is None:
        print('Aviso: node Background não encontrado no World — pulando')
        return

    env_node = nodes.get('RandomEnvTexture')
    mapping_node = nodes.get('RandomEnvMapping')
    coord_node = nodes.get('RandomEnvCoord')

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

    hdri_path = random.choice(hdri_paths)
    if env_node.image is None or env_node.image.filepath != hdri_path:
        env_node.image = bpy.data.images.load(hdri_path, check_existing=True)

    mapping_node.inputs['Rotation'].default_value = (0, 0, random.uniform(0, rotation_range))
    bg_node.inputs['Strength'].default_value = random.uniform(0.4, 1.0)


def randomly_tint_material(material_to_change):
    '''Idêntica à versão do Atividade9 — insere/reaproveita um Mix Color
    (RandomTint) entre a Image Texture e o Base Color.'''
    nodes = material_to_change.node_tree.nodes
    links = material_to_change.node_tree.links

    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        print(f'Aviso: nenhum Principled BSDF em {material_to_change.name}')
        return

    base_color_input = bsdf.inputs['Base Color']
    mix_node = next((n for n in nodes if n.type == 'MIX' and n.name == 'RandomTint'), None)

    if mix_node is None:
        if not base_color_input.is_linked:
            print(f'Aviso: Base Color de {material_to_change.name} não linkado — nada pra tingir')
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
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    mix_node.inputs['B'].default_value = (r, g, b, 1.0)
    mix_node.inputs['Factor'].default_value = random.uniform(0.55, 0.85)


def randomly_orbit_camera(camera_name='Camera', target=(0.4, -0.17, -0.1),
                            distance_range=(2.2, 3.2), elevation_range=(15, 60)):
    '''Sem rig de Follow Path nesta cena (diferente do Atividade9) e sem
    Track To na câmera — posição E rotação são setadas diretamente aqui,
    mesmo padrão que randomly_light_scene já usa pra luz. distance_range
    testado empiricamente (2.0/2.75/3.5m): 2.75m enquadra os dois objetos
    inteiros com boa margem; abaixo de ~2.2m corta as pernas da cadeira,
    acima de ~3.2m sobra espaço vazio demais no quadro.'''
    camera_obj = bpy.context.scene.objects.get(camera_name)
    if camera_obj is None:
        print(f'Aviso: "{camera_name}" não encontrado — pulando órbita de câmera')
        return

    azimuth = random.uniform(0, 2 * math.pi)
    elevation = random.uniform(math.radians(elevation_range[0]), math.radians(elevation_range[1]))
    distance = random.uniform(*distance_range)

    x = target[0] + distance * math.cos(elevation) * math.cos(azimuth)
    y = target[1] + distance * math.cos(elevation) * math.sin(azimuth)
    z = target[2] + distance * math.sin(elevation)
    camera_obj.location = (x, y, z)

    direction = camera_obj.location - Vector(target)
    camera_obj.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()


# ============================================================
# Setup
# ============================================================

scene = bpy.context.scene
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.cycles.samples = 128

hdri_paths = [
    '/Users/maria/Downloads/brown_photostudio_02_8k.hdr',
    '/Users/maria/Downloads/cowboy_town_hall_8k.hdr',
    '/Users/maria/Downloads/glasshouse_interior_8k.hdr',
    '/Users/maria/Downloads/historic_cloister_passage_8k.hdr',
    '/Users/maria/Downloads/relax_inn_seaview_suite_8k.hdr',
]

scene_target = (0.4, -0.17, -0.1)

chair_obj = bpy.data.objects['plastic_monobloc_chair_01']
table_obj = bpy.data.objects['round_wooden_table_01']
chair_base_loc = chair_obj.location.copy()
table_base_loc = table_obj.location.copy()

output_root = Path('/Users/maria/FastCamp/segmentation_dataset')
scratch_dir = Path('/tmp/blender_seg_scratch')
scratch_dir.mkdir(parents=True, exist_ok=True)

ng = scene.compositing_node_group
out_node = ng.nodes['File Output']
out_node.directory = str(scratch_dir) + '/'

# O File Output do compositor nomeia os arquivos como <nome-do-blend><slot>.png
# (ex: "Atividade10MinhaVersãoSegmentation2.png"), não só "<slot>.png" —
# confirmado só depois de rodar (o nome não é óbvio pela API). Segmentation2
# <- ID Mask (index 400 = mesa), Segmentation3 <- ID Mask.001 (index 6000 = cadeira).
blend_stem = Path(bpy.data.filepath).stem
TABLE_MASK_NAME = f'{blend_stem}Segmentation2'
CHAIR_MASK_NAME = f'{blend_stem}Segmentation3'

splits = [('train', 280), ('val', 80), ('test', 40)]
total_count = sum(n for _, n in splits)

for split_name, _ in splits:
    (output_root / 'images' / split_name).mkdir(parents=True, exist_ok=True)
    (output_root / 'raw_masks' / split_name / 'table').mkdir(parents=True, exist_ok=True)
    (output_root / 'raw_masks' / split_name / 'chair').mkdir(parents=True, exist_ok=True)

start_time = time.time()
image_count = 0

for split_name, n_images in splits:
    print(f'Starting split: {split_name} | {n_images} images')
    print('=============================================')

    for idx in range(n_images):
        randomly_rotate_object(chair_obj)
        randomly_rotate_object(table_obj)
        randomly_jitter_position(chair_obj, chair_base_loc)
        randomly_jitter_position(table_obj, table_base_loc)
        randomly_light_scene(target=scene_target)
        randomly_set_environment(hdri_paths)
        randomly_orbit_camera(target=scene_target)

        for obj in (chair_obj, table_obj):
            for slot in obj.material_slots:
                if slot.material is not None:
                    randomly_tint_material(slot.material)

        stem = f'{idx:06d}'
        rgb_path = output_root / 'images' / split_name / f'{stem}.png'
        scene.render.filepath = str(rgb_path)

        bpy.ops.render.render(write_still=True)

        # move as máscaras da pasta de scratch (nome fixo, sobrescrito a
        # cada render) pro destino final numerado
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
