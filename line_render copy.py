
# test_example = true

import time

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import wgpu.backends.rs  # noqa: F401, Select Rust backend
import numpy as np
from sphere import Sphere

# %% Create canvas and device

# Create a canvas to render to
canvas = WgpuCanvas(title="Tammes problem solver")

# Create a wgpu device
adapter = wgpu.request_adapter(canvas=canvas, power_preference="high-performance")
device = adapter.request_device()

# Prepare present context
present_context = canvas.get_context()
render_texture_format = present_context.get_preferred_format(device.adapter)
present_context.configure(device=device, format=render_texture_format)

sph = Sphere(32)

index_data = sph.indices
vertex_data = sph.vertices

# Use numpy to create a struct for the uniform
uniform_dtype = [("transform", "float32", (4, 4))]
uniform_data = np.zeros((), dtype=uniform_dtype)

# Create vertex buffer, and upload data
vertex_buffer = device.create_buffer_with_data(
    data=vertex_data, usage=wgpu.BufferUsage.VERTEX
)

# Create index buffer, and upload data
index_buffer = device.create_buffer_with_data(
    data=index_data, usage=wgpu.BufferUsage.INDEX
)

# Create uniform buffer - data is uploaded each frame
uniform_buffer = device.create_buffer(
    size=uniform_data.nbytes, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
)

shader_source = """
struct Locals {
    transform: mat4x4<f32>;
};
[[group(0), binding(0)]]
var<uniform> r_locals: Locals;

struct VertexInput {
    [[location(0)]] pos : vec3<f32>;
};
struct VertexOutput {
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main(in: VertexInput) -> VertexOutput {
    let r : f32 = in.pos.x;
    let theta : f32 = in.pos.z;
    let phi : f32 = in.pos.y; 
   
   
    let cart: vec3<f32> = vec3<f32>(r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta));
    
    let ndc: vec4<f32> = r_locals.transform * vec4<f32>(cart,1.0);
    var out: VertexOutput;
    out.pos = vec4<f32>(ndc.x, ndc.z,  0.0, 1.0);
    return out;
}


[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return vec4<f32>(1.0,1.0,1.0,1.0);
}
"""

shader = device.create_shader_module(code=shader_source)


# %% The bind groups

# We always have two bind groups, so we can play distributing our
# resources over these two groups in different configurations.
bind_groups_entries = [[]]
bind_groups_layout_entries = [[]]

bind_groups_entries[0].append(
    {
        "binding": 0,
        "resource": {
            "buffer": uniform_buffer,
            "offset": 0,
            "size": uniform_buffer.size,
        },
    }
)
bind_groups_layout_entries[0].append(
    {
        "binding": 0,
        "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
        "buffer": {"type": wgpu.BufferBindingType.uniform},
    }
)

# Create the wgou binding objects
bind_group_layouts = []
bind_groups = []

for entries, layout_entries in zip(bind_groups_entries, bind_groups_layout_entries):
    bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
    bind_group_layouts.append(bind_group_layout)
    bind_groups.append(
        device.create_bind_group(layout=bind_group_layout, entries=entries)
    )

pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)


# %% The render pipeline

render_pipeline = device.create_render_pipeline(
    layout=pipeline_layout,
    vertex={
        "module": shader,
        "entry_point": "vs_main",
        "buffers": [
            {
                "array_stride": 3 * 4,
                "step_mode": wgpu.VertexStepMode.vertex,
                "attributes": [
                    {
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 0,
                    },
                    #{
                    #    "format": wgpu.VertexFormat.float32x2,
                    #    "offset": 4 * 4,
                    #    "shader_location": 1,
                    #},
                ],
            },
        ],
    },
    primitive={
        "topology": wgpu.PrimitiveTopology.line_list,
        #"front_face": wgpu.FrontFace.ccw,
        #"cull_mode": wgpu.CullMode.none,
    },
    depth_stencil=None,
    multisample=None,
    fragment={
        "module": shader,
        "entry_point": "fs_main",
        "targets": [
            {
                "format": render_texture_format,
                "blend": {
                    "alpha": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                    "color": (
                        wgpu.BlendFactor.one,
                        wgpu.BlendFactor.zero,
                        wgpu.BlendOperation.add,
                    ),
                },
            }
        ],
    },
)
sph.check_distances()
sph.update()

# %% Setup the render function


def draw_frame():
    sph.step()
    #sph.update()
    # Update uniform transform
    a1 = 0.11*time.time()
    a2 = 0.3*time.time()
    s = 0.8
    ortho = np.array(
        [
            [s, 0, 0, 0],
            [0, s, 0, 0],
            [0, 0, s, 0],
            [0, 0, 0, 1],
        ],
    )
    rot1 = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(a1), -np.sin(a1), 0],
            [0, np.sin(a1), +np.cos(a1), 0],
            [0, 0, 0, 1],
        ],
    )
    rot2 = np.array(
        [
            [np.cos(a2), 0, np.sin(a2), 0],
            [0, 1, 0, 0],
            [-np.sin(a2), 0, np.cos(a2), 0],
            [0, 0, 0, 1],
        ],
    )

    rot_x = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(a2), -np.sin(a2), 0],
            [0, np.sin(a2), np.cos(a2), 0],
            [0, 0, 0, 1],
        ],
    )

    rot_z = np.array(
        [
           
            [np.cos(a2), -np.sin(a2), 0,0],
            [np.sin(a2), np.cos(a2), 0,0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )
    uniform_data["transform"] = rot1 @ rot2 @ ortho #rot1 @ rot2 @ 

    # Upload the uniform struct
    tmp_buffer = device.create_buffer_with_data(
        data=uniform_data, usage=wgpu.BufferUsage.COPY_SRC
    )
    vertex_buffer = device.create_buffer_with_data(
    data=sph.vertices, usage=wgpu.BufferUsage.VERTEX
    )

    # Create index buffer, and upload data
    index_buffer = device.create_buffer_with_data(
        data=sph.indices, usage=wgpu.BufferUsage.INDEX

    )
    command_encoder = device.create_command_encoder()
    command_encoder.copy_buffer_to_buffer(
        tmp_buffer, 0, uniform_buffer, 0, uniform_data.nbytes
    )

    current_texture_view = present_context.get_current_texture()
    render_pass = command_encoder.begin_render_pass(
        color_attachments=[
            {
                "view": current_texture_view,
                "resolve_target": None,
                "load_value": (0, 0, 0, 1),
                "store_op": wgpu.StoreOp.store,
            }
        ],
    )

    render_pass.set_pipeline(render_pipeline)
    render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
    render_pass.set_vertex_buffer(0, vertex_buffer)
    for bind_group_id, bind_group in enumerate(bind_groups):
        render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
    
    render_pass.draw_indexed(index_data.size, 1, 0, 0, 0)
    render_pass.end_pass()

    device.queue.submit([command_encoder.finish()])
    
    canvas.request_draw()


canvas.request_draw(draw_frame)

if __name__ == "__main__":
    run()
