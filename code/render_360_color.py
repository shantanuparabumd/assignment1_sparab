"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import imageio

from PIL import Image, ImageDraw
import numpy as np

from code.utils import get_device, get_mesh_renderer, load_mesh


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], device=None, camera_rotation = None, camera_translation = None
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_mesh(cow_path)
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    
    z_coordinates=vertices[:,:,2]
    z_min, _ = torch.min(z_coordinates, dim=1)
    z_max, _ = torch.max(z_coordinates, dim=1)
    
    color1 = torch.tensor([0.0, 1.0, 1.0])  # Green & Blue
    color2 = torch.tensor([0.0, 0.1, 0.0])  # Blue

    alpha = (z_coordinates - z_min) / (z_max - z_min)

    alpha=alpha.unsqueeze(-1)
    
    color = alpha[:, None] * color2 + (1 - alpha[:, None]) * color1

    color = color.squeeze(0) 
    
    # textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(color),
    )
    mesh = mesh.to(device)


    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=camera_rotation, T=camera_translation, fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend

def get_camera_poses(distance=3.0,elevation=0.0,number_of_views=36):

    # List to store all the camera poses
    camera_poses=[]

    for i in range(number_of_views):
        # Compute Azimuth
        # Angle created by the projection of the vector from camera line of sight to object with reference vector
        azimuth = (i/number_of_views)*360.0

        # Compute the view transform for the current view
        view_transform = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=distance,
            elev=elevation,
            azim=azimuth,
            degrees="True",
            device="cpu"
        )
        
        
        # Append the view transform to the list
        camera_poses.append(view_transform)

    return camera_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/cow_render_360.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    frames = 36
    my_images = []

    camera_poses = get_camera_poses(distance=3.0,elevation=0.0,number_of_views=frames)
    for pose in camera_poses:
        
        rend = render_cow(cow_path=args.cow_path, image_size=args.image_size,camera_rotation=pose[0],camera_translation=pose[1])
        image = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(image))

    

    imageio.mimsave('images/render_360_color.gif', my_images, duration=0.2)
