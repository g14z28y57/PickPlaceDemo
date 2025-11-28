import bpy
from mathutils import Vector
import os


def look_at(obj, target: Vector):
    direction = target - obj.location
    quat = direction.to_track_quat('-Z', 'Y')
    obj.rotation_euler = quat.to_euler()


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    # 清理残余数据
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block, do_unlink=True)
    for block in bpy.data.images:
        if block.name != "Render Result":
            bpy.data.images.remove(block, do_unlink=True)


# ---------------------- 清空场景 ----------------------
clear_scene()

# ---------------------- 创建蓝色立方体 ----------------------
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0.5))
cube = bpy.context.active_object
cube.name = "BlueCube"

mat_cube = bpy.data.materials.new(name="BlueMaterial")
mat_cube.use_nodes = True
bsdf_cube = mat_cube.node_tree.nodes["Principled BSDF"]
bsdf_cube.inputs['Base Color'].default_value = (0.0, 0.4, 1.0, 1)  # 鲜艳的蓝色
bsdf_cube.inputs['Roughness'].default_value = 0.4
bsdf_cube.inputs['Metallic'].default_value = 1.0
cube.data.materials.append(mat_cube)

# ---------------------- 创建红色夹爪 ----------------------
bpy.ops.mesh.primitive_cube_add(size=1, location=(5, 5, 3))
robot_arm = bpy.context.active_object
robot_arm.name = "RobotArm"
robot_arm.scale = (1, 1, 6)

mat_robot_arm = bpy.data.materials.new(name="RedMaterial")
mat_robot_arm.use_nodes = True
bsdf_robot_arm = mat_robot_arm.node_tree.nodes["Principled BSDF"]
bsdf_robot_arm.inputs['Base Color'].default_value = (1.0, 0.2, 0.2, 1)  # 鲜艳的蓝色
bsdf_robot_arm.inputs['Roughness'].default_value = 0.4
bsdf_robot_arm.inputs['Metallic'].default_value = 1.0
robot_arm.data.materials.append(mat_robot_arm)

# ---------------------- 创建灰色地面 ----------------------
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
floor = bpy.context.active_object
floor.name = "GrayFloor"
floor.scale = (1, 1, 1)  # 做出一个很大的平面

mat_floor = bpy.data.materials.new(name="GrayMaterial")
mat_floor.use_nodes = True
bsdf_floor = mat_floor.node_tree.nodes["Principled BSDF"]
bsdf_floor.inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1)
bsdf_floor.inputs['Roughness'].default_value = 0.6
bsdf_floor.inputs['Metallic'].default_value = 0.0
floor.data.materials.append(mat_floor)

# ---------------------- 创建物体放置区域 ----------------------
bpy.ops.mesh.primitive_plane_add(size=2, location=(8, 8, 0.01))
bin_place = bpy.context.active_object
bin_place.name = "BinPlace"
bin_place.scale = (1, 1, 1)  # 做出一个很大的平面

mat_bin_place = bpy.data.materials.new(name="GreenMaterial")
mat_bin_place.use_nodes = True
bsdf_bin_place = mat_bin_place.node_tree.nodes["Principled BSDF"]
bsdf_bin_place.inputs['Base Color'].default_value = (0.0, 0.9, 0.2, 1)
bsdf_bin_place.inputs['Roughness'].default_value = 0.2
bsdf_bin_place.inputs['Metallic'].default_value = 0.0
bin_place.data.materials.append(mat_bin_place)

# ---------------------- 创建相机并看向原点 ----------------------

bpy.ops.object.camera_add(location=(30, 0, 40))
camera = bpy.context.active_object
camera.name = "MainCamera"
look_at(camera, Vector((0, 0, 0)))  # 稍微看向立方体中心
bpy.context.scene.camera = camera

# ---------------------- 添加光源 ----------------------
# 1. 强阳光（关键光）
bpy.ops.object.light_add(type='SUN', location=(1000, -1000, 1000))
sun = bpy.context.object
sun.data.energy = 2.0
sun.data.angle = 0.526  # 约30度，柔和一点

# 2. 环境光（世界背景）
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.7, 0.8, 1.0, 1)  # 浅蓝灰色环境光
bg.inputs[1].default_value = 0.1  # 强度降低，避免过曝

# 3. 开启 Cycles 自带的接触阴影（让立方体贴地部分更黑，超级真实）
bpy.context.scene.cycles.use_fast_gi = True
bpy.context.scene.render.film_transparent = False

# 可选：加载 HDRI（如果有路径的话）
# env_texture = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
# env_texture.image = bpy.data.images.load("D:/hdri/studio.exr")
# world.node_tree.links.new(env_texture.outputs['Color'], bg.inputs['Color'])

# ---------------------- 渲染设置 ----------------------
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'          # 如果你有支持的显卡
scene.cycles.samples = 64            # 渲染质量（可根据需要调高）
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'

# ---------------------- 输出路径（请改成你自己的） ----------------------
output_path = os.path.join(os.path.dirname(__file__), "ScreenShot.png")
scene.render.filepath = output_path

# ---------------------- 开始渲染 ----------------------
print("开始渲染...")
bpy.ops.render.render(write_still=True)
print(f"渲染完成！图片已保存至：{output_path}")