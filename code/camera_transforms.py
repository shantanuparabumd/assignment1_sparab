"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np

from code.utils import get_device, get_mesh_renderer


def render_cow(
    cow_path="data/cow_with_axis.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)

    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    renderer = get_mesh_renderer(image_size=image_size)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.t().unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow_with_axis.obj")
    parser.add_argument("--image_size", type=int, default=256)
    
    args = parser.parse_args()

    # Define Rotations and Translations

    # Image 1 90 degree anticlockwise Z-axis
    R_relative_1=[[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T_relative_1=[0, 0, 0]

    # Image 2 90 degree anticlockwise Y-axis
    R_relative_2=[[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    T_relative_2=[3, 0, 3]

    # Image 3 Translated along Z
    R_relative_3=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_relative_3=[0, 0, 3]

    # Image 1 Translated along X
    R_relative_4=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    T_relative_4=[0.5, 0, 0]

    # Image 1 Translated along X
    R_relative_5=[[0.7071, -0.7071, 0], [0.7071, 0.7071, 0], [0, 0, 1]]
    T_relative_5=[0, 0, 0]

    plt.imsave("images/transform_cow_1.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size,R_relative=R_relative_1,T_relative=T_relative_1))

    plt.imsave("images/transform_cow_2.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size,R_relative=R_relative_2,T_relative=T_relative_2))

    plt.imsave("images/transform_cow_3.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size,R_relative=R_relative_3,T_relative=T_relative_3))

    plt.imsave("images/transform_cow_4.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size,R_relative=R_relative_4,T_relative=T_relative_4))

    plt.imsave("images/transform_cow_5.jpg", render_cow(cow_path=args.cow_path, image_size=args.image_size,R_relative=R_relative_5,T_relative=T_relative_5))