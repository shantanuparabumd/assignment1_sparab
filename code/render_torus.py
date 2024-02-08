"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

import imageio

from PIL import Image, ImageDraw


from code.utils import get_device, get_mesh_renderer, get_points_renderer




def render_torus(image_size=256, num_samples=200, device=None, camera_rotation = None, camera_translation = None):
    """
    Renders a sphere using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2 * np.pi, num_samples)

    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)

    # Parameters for torus
    R = 3 # Major radius
    r = 1  # Minor radius

    x = (R+r*torch.cos(Theta)) * torch.cos(Phi)
    y = (R+r*torch.cos(Theta)) * torch.sin(Phi)
    z = r*torch.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=camera_rotation, T=camera_translation, fov=60, device=device
    )

    # Place a point light in front of the object.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    renderer = get_points_renderer(image_size=image_size,background_color=(0, 0, 0), device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras,  lights=lights)
    return rend.cpu().numpy()[0, ..., :3] 


def render_torus_mesh(image_size=256, voxel_size=64, device=None, camera_rotation = None, camera_translation = None):
    if device is None:
        device = get_device()

    # Parameters for torus
    R = 0.6 # Major radius
    r = 0.2  # Minor radius

    min_value = -1.1
    max_value = 1.1
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)

    # Implicit Representation of Torus
    voxels = (torch.sqrt(X**2 + Y**2) - R)**2 + Z**2 - r**2

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    
    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=camera_rotation, T=camera_translation, fov=60, device=device
    )

    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)

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
    parser.add_argument(
        "--render",
        type=str,
        default="parametric",
        choices=["parametric", "implicit"],
    )
    
    args = parser.parse_args()
    if args.render == "parametric":
        frames = 36
        my_images = []

        camera_poses = get_camera_poses(distance=9.0,elevation=0.0,number_of_views=frames)
        for pose in camera_poses:
            rend = render_torus(camera_rotation=pose[0],camera_translation=pose[1])
            image = Image.fromarray((rend * 255).astype(np.uint8))
            my_images.append(np.array(image))

        
        imageio.mimsave('images/torus_parametric_render_360.gif', my_images, duration=0.2)

    elif args.render == "implicit":
        frames = 36
        my_images = []

        camera_poses = get_camera_poses(distance=3.0,elevation=0.0,number_of_views=frames)
        for pose in camera_poses:
            rend = render_torus_mesh(camera_rotation=pose[0],camera_translation=pose[1])
            image = Image.fromarray((rend * 255).astype(np.uint8))
            my_images.append(np.array(image))

        
        imageio.mimsave('images/torus_implicit_render_360.gif', my_images, duration=0.2)
    else:
        raise Exception("Did not understand {}".format(args.render))
  
    
    

