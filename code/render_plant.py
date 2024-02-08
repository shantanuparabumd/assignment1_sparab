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
from tqdm.auto import tqdm
from PIL import Image, ImageDraw


from code.utils import get_device, get_mesh_renderer, get_points_renderer,unproject_depth_image


def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_plant_360(
    point_cloud,rgba,path,
    image_size=256,
    background_color=(1, 1, 1),
    device=None
):
    """
    Renders a point cloud.
    """
    if device is None:
        device = get_device()

    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    print(point_cloud.shape)
    verts = torch.Tensor(point_cloud).unsqueeze(0)
    rgba = torch.Tensor(rgba).unsqueeze(0)
    print(verts.shape)
    print(rgba.shape)
    print(rgba[0])
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgba).to(device)
    
    # Place a point light in front of the plant.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]],
                                            device=device)
    frames = 144
    my_images = []

    
    camera_poses = get_camera_poses(distance=7.0,number_of_views=frames)
    for pose in tqdm(camera_poses):
        cameras_360 = pytorch3d.renderer.FoVPerspectiveCameras(
            R=pose[0], T=pose[1], fov=60, device=device
        )
        rend = renderer(point_cloud, cameras=cameras_360, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)  # (B, H, W, 4) -> (H, W, 3)
        image = Image.fromarray((rend * 255).astype(np.uint8))
        my_images.append(np.array(image))

    

    imageio.mimsave(path, my_images, duration=10.0)
    
    



def get_camera_poses(distance=3.0,number_of_views=36):

    # List to store all the camera poses
    camera_poses=[]

    for i in range(number_of_views):
        # Compute Azimuth
        # Angle created by the projection of the vector from camera line of sight to object with reference vector
        azimuth = (i/number_of_views)*360.0

        # Compute the view transform for the current view
        view_transform = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=distance,
            azim=azimuth,
            device="cpu"
        )
        
        # Append the view transform to the list
        camera_poses.append(view_transform)

    return camera_poses


if __name__ == "__main__":
    
    data=load_rgbd_data()

    rgb1=data["rgb1"]
    depth1=data["depth1"]
    mask1=data["mask1"]
    cameras1=data["cameras1"]

    rgb2=data["rgb2"]
    depth2=data["depth2"]
    mask2=data["mask2"]
    cameras2=data["cameras2"]

    point_cloud1,rgba1=unproject_depth_image(torch.tensor(rgb1),torch.tensor(mask1),torch.tensor(depth1),cameras1)

    

    
    point_cloud2,rgba2=unproject_depth_image(torch.tensor(rgb2),torch.tensor(mask2),torch.tensor(depth2),cameras2)

    

    R = torch.tensor([[-1, 0, 0],
                      [0, -1, 0],
                      [0, 0, 1]], dtype=torch.float32)
    point_cloud1 = torch.matmul(point_cloud1, R)
    point_cloud2 = torch.matmul(point_cloud2, R)

    point_cloud3 = torch.cat((point_cloud1, point_cloud2),dim=0)
    rgba3=torch.cat((rgba1, rgba2),dim=0)
    
    print("Rendering Image View 1")
    render_plant_360(point_cloud1,rgba1,path="images/render_360_plant_1.gif")
    print("Rendering Image View 2")
    render_plant_360(point_cloud2,rgba2,path="images/render_360_plant_2.gif")
    print("Rendering Combined Image View")
    render_plant_360(point_cloud3,rgba3,path="images/render_360_plant_3.gif")


    
    
    
    
    

