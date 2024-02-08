"""
Usage:
    python -m starter.dolly_zoom --num_frames 10
"""

import argparse

import imageio
import numpy as np
import pytorch3d
import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from code.utils import get_device, get_mesh_renderer


def dolly_zoom(
    image_size=256,
    num_frames=10,
    duration=3.0,
    device=None,
    output_file="images/dolly.gif",
):
    if device is None:
        device = get_device()

    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"])
    mesh = mesh.to(device)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=device)

    fovs = torch.linspace(5, 120, num_frames)

    renders = []
    for fov in tqdm(fovs):
        width=4.5

        distance = width/(2*np.tan(0.5*np.radians(fov)))
        # print("Distance:",distance,"FOV:",fov)
        T = [[0, 0, distance]]  # TODO: Change this.
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(fov=fov, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        renders.append(rend)

    images = []
    for i, r in enumerate(renders):
        image = Image.fromarray((r * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.text((20, 20), f"fov: {fovs[i]:.2f}", fill=(255, 0, 0))
        images.append(np.array(image))
    imageio.mimsave(output_file, images,  duration=duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output_file", type=str, default="images/dolly.gif")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    dolly_zoom(
        image_size=args.image_size,
        num_frames=args.num_frames,
        duration=args.duration,
        output_file=args.output_file,
    )
