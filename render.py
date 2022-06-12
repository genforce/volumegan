# python3.7
"""A simple tool to synthesize images with pre-trained models."""

import os
import math
import click
import random

import skvideo.io
import mcubes
import trimesh
import mrcfile
from tqdm import tqdm
import torch
import numpy as np

from configs import build_config, CONFIG_POOL
from models import build_model

from utils.parsing_utils import parse_bool, DictAction
from utils.visualizers import HtmlVisualizer
from utils.image_utils import save_image, load_image, resize_image

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=0.3):
    """Create coordinates of the canonical voxel"""
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples.unsqueeze(0), voxel_origin, voxel_size

def postprocess(images):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`."""
    images = images.detach().cpu().numpy()
    images = (images + 1) * 255 / 2
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images

def preprocess(images):
    """Pre-process images from `numpy array` to `torch tensor`"""
    images = torch.from_numpy(images.astype(np.float32)).cuda() 
    images = images*2.0/255.0 - 1.0
    images = images.permute(0, 3, 1, 2) 
    return images


@click.group(name='Render Script',
             help='Render image, video, shape',
             context_settings={'show_default': True, 'max_content_width': 180})
@click.option('--checkpoint', type=str,
              help='Path to the checkpoint to load.')
@click.option('--work_dir', type=str, default='work_dirs/synthesis',
              help='Directory to save the results. If not specified, '
                   'the results will be saved to '
                   '`work_dirs/synthesis/` by default.')
@click.option('--num', type=int, default=100,
              help='Number of samples to synthesize.')
@click.option('--batch_size', type=int, default=1,
              help='Batch size.')
@click.option('--step', type=int, default=70,
              help='render video steps')
@click.option('--save_raw_synthesis', type=parse_bool, default=False,
              help='Whether to save raw synthesis.')
@click.option('--seed', type=int, default=0,
              help='Seed for sampling.')
@click.option('--render_mode', type=str, default='video',
              help='choose the type of the render results')
@click.option('--trajectory_type', type=str, default='orbit',
              help='choose the type of the rendering trajectory')
@click.option('--generate_html', type=parse_bool, default=True,
              help='Whether to generate html.')
def command_group(checkpoint, work_dir, num, batch_size, step, save_raw_synthesis, seed, render_mode, trajectory_type, generate_html):  # pylint: disable=unused-argument
    """Defines a command group for rendering script.

    This function is mainly inherited train.py.
    """


@command_group.result_callback()
@click.pass_context
def main(ctx, kwargs, checkpoint, num, batch_size, step, save_raw_synthesis, seed, render_mode, work_dir, trajectory_type, generate_html):

    config = build_config(ctx.invoked_subcommand, kwargs).get_config()

    # CUDNN settings.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load
    state = torch.load(checkpoint, map_location='cpu')
    # import ipdb;ipdb.set_trace()
    G = build_model(**state['model_kwargs_init']['generator_smooth'])
    # G = build_model(**config.models.generator.model)
    G.load_state_dict(state['models']['generator_smooth'], strict=True)
    G.eval().cuda()
    G_kwargs= dict(noise_mode='const',
                   fused_modulate=False,
                   impl='cuda',
                   fp16_res=None)

    os.makedirs(work_dir, exist_ok=True)
    job_name = f'{ctx.invoked_subcommand}_{num}'

    # Sample and synthesize.
    if render_mode == 'shape':
        print(f'Synthesizing {num} shapes ...')
        assert batch_size == 1, 'batch_size should be set as 1 when render_mode is shape!'

        # set default value
        voxel_res = 256
        voxel_org = [0,0,0]
        voxel_len = 0.3
        # num points per batch
        max_batch = 100000

        # shape_path
        shapes_path = os.path.join(work_dir, 'shapes')
        os.makedirs(shapes_path, exist_ok=True)

        indices = list(range(num))
        for batch_idx in tqdm(range(0, num, batch_size), leave=False):
            sub_indices = indices[batch_idx:batch_idx + batch_size]
            code = torch.randn(len(sub_indices), G.z_dim).cuda()
            with torch.no_grad():
                for ci in range(len(sub_indices)):
                    wp = G.mapping(code[ci:ci+1], None)['wp']
                    samples, voxel_origin, voxel_size = create_samples(voxel_res, voxel_org, voxel_len)
                    samples = samples.to(code.device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1, 1, 1), device=code.device)
                    head = 0
                    while head < samples.shape[1]:
                        fg_points = samples[:, head:head+max_batch].unsqueeze(2).unsqueeze(2)
                        fg_density = G.nerfmlp(wp, fg_points, None, )['sigma']
                        sigmas[:, head:head+max_batch] = fg_density
                        head += max_batch

                    # save mrc file
                    voxel_grid = sigmas.reshape((voxel_res, voxel_res, voxel_res)).cpu().numpy()
                    with mrcfile.new_mmap(f'{shapes_path}/{sub_indices[ci]:06d}.mrc',
                                          overwrite=True,
                                          shape=voxel_grid.shape,
                                          mrc_mode=2) as mrc:
                        mrc.data[:] = voxel_grid

                    # save ply file
                    vs, fs = mcubes.marching_cubes(voxel_grid, 10)
                    vs = voxel_origin + vs*voxel_size
                    mesh = trimesh.Trimesh(vertices=vs, faces=fs)
                    _ = mesh.export(f'{shapes_path}/{sub_indices[ci]:06d}.ply')
        print(f'Finish synthesizing {num} shapes.')

    if render_mode == 'video':
        print(f'Synthesizing {num} videos...')
        # video_path
        videos_path = os.path.join(work_dir, 'videos')
        os.makedirs(videos_path, exist_ok=True)

        if generate_html:
            html = HtmlVisualizer(num_rows=num, num_cols=step)
        ps_cfg = state['model_kwargs_init']['generator']['ps_cfg']
        h_mean, v_mean = ps_cfg['horizontal_mean'], ps_cfg['vertical_mean']
        default_fov = ps_cfg['fov']

        trajectory = []
        for t in np.linspace(0, 1, step):
            if trajectory_type == 'orbit':
                pitch = 0.2 * np.cos(t * 2 * math.pi) + v_mean
                yaw = 0.4 * np.sin(t * 2 * math.pi) + h_mean
                fov = default_fov
            elif trajectory_type == 'front':
                pitch = v_mean
                yaw = -math.pi/4*(t-1/2) + h_mean
                fov = default_fov
            else:
                raise NotImplemenedError
            trajectory.append((pitch, yaw, fov))

        code = torch.randn(num, G.z_dim).cuda()
        ps_kwargs = {}
        indices = list(range(num))
        for batch_idx in tqdm(range(0, num, batch_size), leave=False):
            sub_indices = indices[batch_idx:batch_idx + batch_size]
            sub_code = code[sub_indices]
            sub_frames = [[] for i in sub_indices]
            with torch.no_grad():
                for tidx,(pitch, yaw, fov) in tqdm(enumerate(trajectory), leave=False):
                    G_kwargs['trunc_psi'] = 0.7 
                    G_kwargs['trunc_layers'] = 8 
                    ps_kwargs['horizontal_stddev'] = 0
                    ps_kwargs['vertical_stddev'] = 0
                    ps_kwargs['num_steps'] = ps_cfg['num_steps']*3
                    ps_kwargs['horizontal_mean'] = yaw
                    ps_kwargs['vertical_mean'] = pitch

                    images = G(sub_code, ps_kwargs=ps_kwargs, **G_kwargs)['image']
                    images = postprocess(images)
                    for sidx, (sub_frame, image) in enumerate(zip(sub_frames, images)):
                        sub_frame.append(image)
                        if generate_html:
                            html.set_cell(sub_indices[sidx], tidx, image=image)
                # import ipdb;ipdb.set_trace()
                for sub_idx, sub_frame in zip(sub_indices, sub_frames): 
                    os.makedirs(os.path.join(work_dir, 'videos'), exist_ok=True)
                    writer = skvideo.io.FFmpegWriter(f'{videos_path}/{sub_idx:06d}.mp4', outputdict={'-pix_fmt': 'yuv420p', '-crf': '21'})
                    for f in sub_frame:
                        writer.writeFrame(f)
                    writer.close()
        if generate_html:
            html.save(os.path.join(work_dir, f'{job_name}_images.html'))
        print(f'Finish synthesizing {num} videos.')

if __name__ == '__main__':
    # Append all available commands (from `configs/`) into the command group.
    for cfg in CONFIG_POOL:
        command_group.add_command(cfg.get_command())
    # Run by interacting with command line.
    command_group()  # pylint: disable=no-value-for-parameter
