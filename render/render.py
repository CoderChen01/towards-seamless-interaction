import argparse
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple

import av
import cv2
import imageio
import librosa
import numpy as np
import soundfile as sf
import torch
import trimesh

# project-specific
from config import get_cfg_defaults
from flame.FLAME_fake import FLAME
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from scipy.signal import savgol_filter
from tqdm import tqdm

if platform.system() == "Linux":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # harmless for PyTorch3D


# ===============================
# Audio utils
# ===============================


def align_audio_to_duration(
    wav_path: str,
    target_duration: float,
    tmp_dir: str,
    sr: int = 48000,
) -> str:
    """Resample to sr and trim/pad to exactly target_duration seconds. Return temp wav path.
    If input wav is missing, generate silence.
    """
    tmp_wav = str(Path(tmp_dir) / Path(wav_path).name)

    if not os.path.exists(wav_path):
        y = np.zeros(int(round(target_duration * sr)), dtype=np.float32)
        # tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(tmp_wav, y, sr)
        return tmp_wav

    y, src_sr = librosa.load(wav_path, sr=None, mono=True)
    if src_sr != sr:
        y = librosa.resample(y, orig_sr=src_sr, target_sr=sr)
    tgt_len = int(round(target_duration * sr))
    if len(y) >= tgt_len:
        y = y[:tgt_len]
    else:
        y = np.pad(y, (0, tgt_len - len(y)), mode="constant")
    # tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    sf.write(tmp_wav, y, sr)
    return tmp_wav


# ===============================
# Camera/Renderer
# ===============================


# @dataclass
# class CameraIntrinsics:
#     fx: float
#     fy: float
#     cx: float
#     cy: float
#     width: int
#     height: int
#     znear: float = 0.01
#     zfar: float = 3.0


def build_renderer(device: torch.device, image_size: Tuple[int, int] = (512, 512)):
    R, T = look_at_view_transform(0.6, 0, 0)
    cameras = FoVPerspectiveCameras(
        device=device, R=R.to(device), T=T.to(device), fov=30.0
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
        perspective_correct=True,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer, cameras, image_size


def make_meshes(
    verts_bxvx3: torch.Tensor, faces_fx3: torch.Tensor, white: bool = True
) -> Meshes:
    """Build Meshes with a neutral '高级灰' vertex color by default.
    white flag is kept for API compatibility but maps to gray to match the requested style.
    """
    B, V, _ = verts_bxvx3.shape
    if faces_fx3.dim() == 2:
        faces_bxfx3 = faces_fx3.unsqueeze(0).expand(B, -1, -1).to(verts_bxvx3.device)
    else:
        faces_bxfx3 = faces_fx3.to(verts_bxvx3.device)

    gray = 0.72
    color = torch.tensor(
        [gray, gray, gray], device=verts_bxvx3.device, dtype=verts_bxvx3.dtype
    )
    verts_rgb = color.view(1, 1, 3).expand(B, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    meshes = Meshes(
        verts=[verts_bxvx3[i] for i in range(B)],
        faces=[faces_bxfx3[i] for i in range(B)],
        textures=textures,
    )
    return meshes


def render_batch(
    renderer: MeshRenderer,
    cameras: FoVPerspectiveCameras,
    verts_bxvx3: torch.Tensor,
    faces_fx3: torch.Tensor,
    white_threshold: int = 254,
    do_gamma: bool = True,
) -> np.ndarray:

    meshes = make_meshes(verts_bxvx3, faces_fx3, white=True)

    blend = BlendParams(background_color=(1.0, 1.0, 1.0))
    rgb = renderer(meshes, cameras=cameras, blend_params=blend)[
        ..., :3
    ]  # (B,H,W,3), float32 [0,1]
    rgb = rgb.clamp_(0, 1)

    if do_gamma:
        rgb = rgb.pow_(1.0 / 2.2)

    rgb_u8 = (rgb * 255.0 + 0.5).to(torch.uint8)  # (B,H,W,3) uint8

    white_mask = (rgb_u8 > white_threshold).all(dim=-1)  # (B,H,W) bool
    alpha = (~white_mask).to(torch.uint8).mul_(255).unsqueeze(-1)  # (B,H,W,1) uint8

    rgba = torch.cat([rgb_u8, alpha], dim=-1)  # (B,H,W,4) uint8
    return rgba.detach().cpu().numpy()


def _add_label_opencv(
    img: np.ndarray, text: str, margin_px: int = 8, font_scale: float = 1.0
) -> np.ndarray:
    """
    Draw text directly on the top of RGB numpy image using OpenCV.
    Adds a white background rectangle behind the text.
    """
    # OpenCV uses BGR, but our image is RGB, so flip channels
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color_text = (0, 0, 0)  # black text
    color_bg = (255, 255, 255)  # white background
    thickness = 2

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Draw background rectangle
    cv2.rectangle(
        img_bgr, (0, 0), (img_bgr.shape[1], text_h + margin_px * 2), color_bg, -1
    )
    # Draw text (centered horizontally)
    x = (img_bgr.shape[1] - text_w) // 2
    y = text_h + margin_px // 2
    cv2.putText(
        img_bgr,
        text,
        (x, y),
        font,
        font_scale,
        color_text,
        thickness,
        lineType=cv2.LINE_AA,
    )

    # Convert back to RGB
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def render_sequence_single_gpu(
    verts_txvx3: torch.Tensor,
    faces_fx3: torch.Tensor,
    image_size: Tuple[int, int],
    batch_size: int,
    device: torch.device,
    show_tqdm: bool,
    label: str,
    *,
    frame_start: int = 0,
):
    """
    Render a sequence on single GPU and return (raw_frames, labeled_frames).
    - Text '{label} #{global_frame_idx}' is drawn directly on top via OpenCV.
    - RGB images, fast (no PIL).
    """
    renderer, cameras, (H, W) = build_renderer(device, image_size)
    raw_frames, labeled_frames = [], []

    rng = range(0, verts_txvx3.shape[0], batch_size)
    iterator = tqdm(rng, desc=f"render@{device}") if show_tqdm else rng

    frame_idx = frame_start
    for s in iterator:
        e = min(s + batch_size, verts_txvx3.shape[0])
        batch = verts_txvx3[s:e].to(device)
        imgs = render_batch(renderer, cameras, batch, faces_fx3.to(device))

        for img in imgs:
            raw_frames.append(img)
            # ensure uint8 RGB
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 1)
                img = (img * 255 + 0.5).astype(np.uint8)
            text = f"{label} #{frame_idx}"
            labeled = _add_label_opencv(img, text)
            labeled_frames.append(labeled)
            frame_idx += 1

    return np.stack(raw_frames, axis=0), np.stack(labeled_frames, axis=0)


def render_and_mux(
    verts_txvx3: np.ndarray,
    faces_fx3: np.ndarray,
    tmp_path: str,
    label: str,
    wav_path: Optional[str] = None,
    image_size: Tuple[int, int] = (512, 512),
    fps: int = 25,
    device: Optional[torch.device] = None,
    batch_size: int = 256,
    show_tqdm: bool = True,
):
    os.makedirs(tmp_path, exist_ok=True)
    tmp_image_path = os.path.join(tmp_path, "frames")
    os.makedirs(tmp_image_path, exist_ok=True)

    tmp_mp4 = os.path.join(tmp_path, "tmp.mp4")

    device = device if device else torch.device("cpu")
    raw_frames, labeled_frames = render_sequence_single_gpu(
        torch.from_numpy(verts_txvx3).float(),
        torch.from_numpy(faces_fx3).long(),
        image_size,
        batch_size=batch_size,
        device=device,
        show_tqdm=show_tqdm,
        label=label,
    )
    if not os.path.exists(tmp_mp4):
        for k in range(len(raw_frames)):
            imageio.imwrite(os.path.join(tmp_image_path, f"{k:03d}.png"), raw_frames[k])
        imageio.mimsave(tmp_mp4, labeled_frames, fps=fps)  # type: ignore

    adj_wav = None
    if wav_path is not None:
        T = verts_txvx3.shape[0]
        target_dur = T / float(fps)
        adj_wav = align_audio_to_duration(wav_path, target_dur, tmp_path)

    return labeled_frames, adj_wav


def mix_wavs(wav_paths: List[str], tmp_dir: str, output_name: str = "mixed.wav") -> str:
    """
    Mix multiple WAV files and save the mixed audio to tmp_dir.

    Args:
        wav_paths (List[str]): List of input wav file paths.
        tmp_dir (str): Directory to save the mixed wav.
        output_name (str): Output filename (default 'mixed.wav').

    Returns:
        str: Full path to the mixed wav file.
    """
    if not wav_paths:
        raise ValueError("wav_paths cannot be empty.")

    os.makedirs(tmp_dir, exist_ok=True)

    # Step 1: load all wavs (use first as base)
    mixed = None
    for path in wav_paths:
        try:
            audio = AudioSegment.from_wav(path)
        except Exception:
            print(f"[WARN] Skipping invalid or unreadable audio: {path}")
            continue

        if mixed is None:
            mixed = audio
        else:
            # Overlay (shorter one is padded automatically)
            mixed = mixed.overlay(audio)

    if mixed is None:
        raise RuntimeError("No valid audio found to mix.")

    # Step 2: export to tmp_dir
    output_path = os.path.join(tmp_dir, output_name)
    mixed.export(output_path, format="wav")
    return output_path


def concat_frames_horizontally(labeled_frames: list[np.ndarray]) -> np.ndarray:
    """
    Concatenate a list of RGB images horizontally (left to right).
    Works even if each frame shape is (H, W, 3, 1).
    """
    if not labeled_frames:
        raise ValueError("labeled_frames cannot be empty.")

    # normalize dimensions
    labeled_frames = [np.squeeze(f) for f in labeled_frames]  # remove trailing 1

    # ensure consistent size
    h_list = [f.shape[0] for f in labeled_frames]
    w_list = [f.shape[1] for f in labeled_frames]
    if len(set(h_list)) != 1:
        raise ValueError("All frames must have same height for horizontal concat.")

    # concatenate along width axis (axis=2)
    merged = np.concatenate(labeled_frames, axis=2)
    return merged


def export_video_with_audio(frames, audio_path: str, output_path: str, fps: int = 25):

    # --- normalize frames ---
    if isinstance(frames, list):
        frames = np.stack(frames)
    frames = np.asarray(frames)
    if frames.ndim > 4:
        frames = np.squeeze(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape (T,H,W,3), got {frames.shape}")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    num_frames = frames.shape[0]

    # === open MP4 container ===
    container = av.open(
        output_path, mode="w", format="mp4", options={"movflags": "+faststart"}
    )

    # --- video stream ---
    stream_v = container.add_stream("libx264", rate=fps)
    stream_v.width = frames.shape[2]
    stream_v.height = frames.shape[1]
    stream_v.pix_fmt = "yuv420p"
    stream_v.options = {
        "preset": "ultrafast",
        "tune": "zerolatency",
        "profile": "baseline",
    }

    # --- audio ---
    audio_seg = AudioSegment.from_file(audio_path)
    audio_seg = audio_seg.set_channels(2).set_frame_rate(44100)
    samples = np.array(audio_seg.get_array_of_samples())
    samples = samples.reshape((-1, audio_seg.channels)).T.copy()
    layout = "stereo"
    sample_rate = 44100

    # try mp3, fallback to aac
    try:
        stream_a = container.add_stream("libmp3lame", rate=sample_rate)
    except av.AVError:
        print("⚠️ libmp3lame not supported, falling back to AAC")
        stream_a = container.add_stream("aac", rate=sample_rate)
    stream_a.layout = layout
    stream_a.bit_rate = 128_000

    # === encode video ===
    for img in frames:
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for pkt in stream_v.encode(frame):
            container.mux(pkt)
    for pkt in stream_v.encode():
        container.mux(pkt)

    # === encode audio ===
    frame_size = 1152
    for i in range(0, samples.shape[1], frame_size):
        chunk = samples[:, i : i + frame_size].copy()
        if chunk.shape[1] == 0:
            continue
        aframe = av.AudioFrame.from_ndarray(chunk, format="s16p", layout=layout)
        aframe.sample_rate = sample_rate
        for pkt in stream_a.encode(aframe):
            container.mux(pkt)

    # flush
    for pkt in stream_a.encode():
        container.mux(pkt)

    container.close()


# ===============================
# Task-specific data path → vertices
# ===============================


def load_faces_from_ply(ply_path: str) -> np.ndarray:
    mesh = trimesh.load(ply_path, process=False)
    faces = np.asarray(mesh.faces, dtype=np.int64)  # type: ignore
    return faces


def pick_device(flag: str) -> torch.device:
    if flag == "cpu":
        return torch.device("cpu")
    if flag.startswith("cuda"):
        return torch.device(flag)
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


class FlameSimpleSG:
    """
    Simplified Savitzky–Golay smoother for streaming FLAME parameters.
    Each call uses the previous 1 s + current 1 s of data (≈ 50 frames total)
    and returns the smoothed result for the current 1 s.
    """

    def __init__(self, dim=56, window=7, poly=2):
        """
        Args:
            dim: number of FLAME parameters
            window: SG filter window length (must be odd)
            poly: polynomial order (< window)
        """
        assert window % 2 == 1, "window must be odd"
        assert poly < window, "poly must be smaller than window length"

        self.dim = dim
        self.window = window
        self.poly = poly
        self.prev = np.zeros((0, dim))  # store last 1 s (25 frames)

    # ------------------------------------------------------------------

    def step(self, chunk):
        """
        Smooth current 1 s chunk (shape [25, 56]) using
        previous + current 2 s window (≈ 50 frames total).
        Returns smoothed output for current 1 s.
        """
        chunk = np.asarray(chunk)
        # combine previous and current data
        data = np.vstack([self.prev, chunk]) if len(self.prev) else chunk

        # apply SG filter along time axis
        smooth = savgol_filter(
            data, window_length=self.window, polyorder=self.poly, axis=0, mode="interp"
        )

        # output only the last 1 s portion
        out = smooth[-len(chunk) :]

        # store current chunk as next round's "previous"
        self.prev = chunk.copy()
        return out

    # ------------------------------------------------------------------

    def flush(self):
        self.prev = np.zeros((0, self.dim))


def get_exp_jaw_neck(data, device, is_smooth=True):
    npy = data
    fps = 25
    chunks = [data[i : i + fps] for i in range(0, len(data), fps)]
    smoother = FlameSimpleSG(dim=3, window=7, poly=2)
    results = [smoother.step(chunk) for chunk in chunks]
    npy = np.vstack(results)
    # if is_smooth:

    # npy = savgol_filter(data, 5, 3, axis=0)
    exp = npy[:, 0:50]
    exp = np.hstack((exp, np.zeros((exp.shape[0], 50))))
    exp = torch.from_numpy(exp).float().to(device)

    jaw = torch.from_numpy(npy[:, 50:53]).float()
    jaw = matrix_to_rotation_6d(axis_angle_to_matrix(jaw)).to(device)

    neck = npy[:, 53:56]
    neck = torch.from_numpy(neck).float()
    neck = matrix_to_rotation_6d(axis_angle_to_matrix(neck)).to(device)

    return exp, jaw, neck


def build_vertices_from_param_seq(
    agent_pred_npy_paths: List[str],
    gt_npz_path: str,
    user_gt_npz_path,
    flame: FLAME,
    device: torch.device,
):

    agent_gt_param = np.load(gt_npz_path)
    user_gt_param = np.load(user_gt_npz_path)

    user_gt_data = np.concat(
        [
            user_gt_param["exp"],
            user_gt_param["pose"][:, 3:],
            user_gt_param["pose"][:, :3],
        ],
        axis=-1,
    )
    user_gt_exp, user_gt_jaw, user_gt_neck = get_exp_jaw_neck(
        user_gt_data, device, False
    )

    agent_gt_data = np.concat(
        [
            agent_gt_param["exp"],
            agent_gt_param["pose"][:, 3:],
            agent_gt_param["pose"][:, :3],
        ],
        axis=-1,
    )
    agent_gt_exp, agent_gt_jaw, agent_gt_neck = get_exp_jaw_neck(
        agent_gt_data, device, False
    )

    agent_pred_exps = []
    agent_pred_jaws = []
    agent_pred_necks = []
    for npy_path in agent_pred_npy_paths:
        agent_data = np.load(npy_path)
        agent_exp, agent_jaw, agent_neck = get_exp_jaw_neck(agent_data, device)
        agent_pred_exps.append(agent_exp)
        agent_pred_jaws.append(agent_jaw)
        agent_pred_necks.append(agent_neck)

    min_length = min(
        [
            *[i.shape[0] for i in agent_pred_exps],
            *[i.shape[0] for i in agent_pred_jaws],
            *[i.shape[0] for i in agent_pred_necks],
            agent_gt_exp.shape[0],
            agent_gt_jaw.shape[0],
            agent_gt_neck.shape[0],
            user_gt_exp.shape[0],
            user_gt_jaw.shape[0],
            user_gt_neck.shape[0],
        ]
    )
    agent_pred_exps = [i[:min_length, :] for i in agent_pred_exps]
    agent_pred_jaws = [i[:min_length, :] for i in agent_pred_jaws]
    agent_pred_necks = [i[:min_length, :] for i in agent_pred_necks]
    agent_gt_exp = agent_gt_exp[:min_length, :]
    agent_gt_jaw = agent_gt_jaw[:min_length, :]
    agent_gt_neck = agent_gt_neck[:min_length, :]
    user_gt_exp = user_gt_exp[:min_length, :]
    user_gt_jaw = user_gt_jaw[:min_length, :]
    user_gt_neck = user_gt_neck[:min_length, :]

    user_gt_shape = torch.from_numpy(agent_gt_param["shape"]).float().to(device)
    user_gt_shape = user_gt_shape[0, :].unsqueeze(0)
    user_gt_shape = torch.cat(
        (user_gt_shape, torch.zeros(1, 200, device=device)), dim=1
    )
    user_gt_shape_params = user_gt_shape.repeat(user_gt_exp.shape[0], 1)

    agent_gt_shape = torch.from_numpy(agent_gt_param["shape"]).float().to(device)
    agent_gt_shape = agent_gt_shape[0, :].unsqueeze(0)
    agent_gt_shape = torch.cat(
        (agent_gt_shape, torch.zeros(1, 200, device=device)), dim=1
    )
    agent_gt_shape_params = agent_gt_shape.repeat(user_gt_exp.shape[0], 1)

    all_agent_pred_vertices = []
    with torch.no_grad():
        for agent_exp, agent_neck, agent_jaw in zip(
            agent_pred_exps, agent_pred_necks, agent_pred_jaws
        ):
            agent_vertices, _, _ = flame(
                shape_params=agent_gt_shape_params,
                expression_params=agent_exp,
                neck_pose_params=agent_neck,
                eye_pose_params=None,
                jaw_pose_params=agent_jaw,
                eyelid_params=None,
            )
            all_agent_pred_vertices.append(agent_vertices.detach().cpu().numpy())
        agent_gt_vertices, _, _ = flame(
            shape_params=agent_gt_shape_params,
            expression_params=agent_gt_exp,
            neck_pose_params=agent_gt_neck,
            eye_pose_params=None,
            jaw_pose_params=agent_gt_jaw,
            eyelid_params=None,
        )
        user_gt_vertices, _, _ = flame(
            shape_params=user_gt_shape_params,
            expression_params=user_gt_exp,
            neck_pose_params=user_gt_neck,
            eye_pose_params=None,
            jaw_pose_params=user_gt_jaw,
            eyelid_params=None,
        )

    agent_neck_mean = np.mean(agent_gt_param["pose"][:, :3], axis=0)
    user_neck_mean = np.mean(user_gt_param["pose"][:, :3], axis=0)

    return {
        "agent_preds": all_agent_pred_vertices,
        "agent_gt": agent_gt_vertices.detach().cpu().numpy(),
        "user_gt": user_gt_vertices.detach().cpu().numpy(),
        "agent_neck_mean": agent_neck_mean,
        "user_neck_mean": user_neck_mean,
    }


# ===============================
# Main
# ===============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interest_of_sample_list", type=str, nargs="+", default=None)
    parser.add_argument("--timar_pred_path", type=str, required=True)
    parser.add_argument("--gt_path", type=str, default="./data/test/", required=True)
    parser.add_argument(
        "--output_path", type=str, default="./render_result/", required=True
    )
    parser.add_argument("--comp_paths", type=str, nargs="+", default=None)
    parser.add_argument("--comp_labels", type=str, nargs="+", default=None)
    parser.add_argument("--template_ply", type=str, default="./flame/FLAME_sample.ply")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--use_tqdm", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device for forward: auto|cpu|cuda:N",
    )
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    # Prepare intrinsics (matches your pyrender setup; adjust if using BIWI scale)

    faces = load_faces_from_ply(args.template_ply)

    device = pick_device(args.device)
    cfg = get_cfg_defaults()

    succ = False
    while not succ:
        try:
            flame = FLAME(cfg).to(device)
            succ = True
        except Exception as e:
            print(f"[ERROR] Initializing FLAME failed with exception: {e}. Retrying...")
            succ = False

    print("Finish initializing FLAME.")
    if args.interest_of_sample_list is not None:
        name_list = args.interest_of_sample_list
    else:
        name_list = os.listdir(args.timar_pred_path)

    print(f"All Name List: {','.join(name_list)}")

    for name in name_list:
        if not name.endswith(".npy"):
            continue

        succ = False
        while not succ:
            try:
                output_path = os.path.join(args.output_path, name.replace(".npy", ""))
                os.makedirs(output_path, exist_ok=True)

                if (
                    os.path.exists(os.path.join(output_path, "render_done.flag"))
                    and not args.force
                ):
                    print(
                        f"Skip {name} since output video exists. Use --force to override."
                    )
                    succ = True
                    continue

                print(f"Start to render {name}")
                agent_wav_path = os.path.join(
                    args.gt_path, name.replace(".npy", ".wav")
                )
                shutil.copy(
                    os.path.join(args.gt_path, name.replace(".npy", ".txt")),
                    os.path.join(output_path, "agent.txt"),
                )
                timar_agent_npy_path = os.path.join(args.timar_pred_path, name)

                agent_gt_npz_path = os.path.join(
                    args.gt_path, name.replace(".npy", ".npz")
                )
                agent_gt_out_tmp_path = os.path.join(
                    output_path, name.replace(".npy", "_agent_gt_out_tmp")
                )
                if name.endswith("speaker1.npy"):
                    user_wav_path = os.path.join(
                        args.gt_path, name.replace("speaker1.npy", "speaker2.wav")
                    )
                    user_gt_npz_path = os.path.join(
                        args.gt_path, name.replace("speaker1.npy", "speaker2.npz")
                    )
                    shutil.copy(
                        os.path.join(
                            args.gt_path, name.replace("speaker1.npy", "speaker2.txt")
                        ),
                        os.path.join(output_path, "user.txt"),
                    )
                else:
                    user_wav_path = os.path.join(
                        args.gt_path, name.replace("speaker2.npy", "speaker1.wav")
                    )
                    user_gt_npz_path = os.path.join(
                        args.gt_path, name.replace("speaker2.npy", "speaker1.npz")
                    )
                    shutil.copy(
                        os.path.join(
                            args.gt_path, name.replace("speaker2.npy", "speaker1.txt")
                        ),
                        os.path.join(output_path, "user.txt"),
                    )
                user_gt_out_tmp = os.path.join(
                    output_path, name.replace(".npy", "_user_gt_out_tmp")
                )

                assert os.path.exists(timar_agent_npy_path)
                assert os.path.exists(agent_gt_npz_path)
                assert os.path.exists(agent_wav_path)
                assert os.path.exists(user_gt_npz_path)
                assert os.path.exists(user_wav_path)

                agent_pred_paths = [timar_agent_npy_path]
                agent_pred_labels = ["TIMAR Agent"]
                agent_pred_out_tmp_paths = [
                    os.path.join(
                        output_path, name.replace(".npy", "_timar_agent_out_tmp")
                    )
                ]
                if args.comp_paths is not None:
                    assert args.comp_labels is not None and len(args.comp_paths) == len(
                        args.comp_labels
                    )
                    for i, p in enumerate(args.comp_paths):
                        comp_path = os.path.join(p, name)
                        assert os.path.exists(comp_path)
                        agent_pred_paths.append(comp_path)
                        curr_label = args.comp_labels[i]
                        agent_pred_labels.append(curr_label)
                        agent_pred_out_tmp_paths.append(
                            os.path.join(
                                output_path,
                                name.replace(
                                    ".npy",
                                    f"_{curr_label.lower().replace(' ', '_')}_out_tmp",
                                ),
                            )
                        )

                all_verts = build_vertices_from_param_seq(
                    agent_pred_paths, agent_gt_npz_path, user_gt_npz_path, flame, device
                )
                print("Finish FLAME forword.")

                agent_adj_wav = None
                all_agent_pred_frames = []
                for i, agent_verts in enumerate(all_verts["agent_preds"]):
                    labeled_frames, adj_wav = render_and_mux(
                        verts_txvx3=agent_verts,
                        faces_fx3=faces,
                        wav_path=agent_wav_path if i == 0 else None,
                        tmp_path=agent_pred_out_tmp_paths[i],
                        image_size=(512, 512),
                        fps=args.fps,
                        device=device,
                        batch_size=args.batch_size,
                        show_tqdm=args.use_tqdm,
                        label=agent_pred_labels[i],
                    )
                    if adj_wav is not None:
                        agent_adj_wav = adj_wav
                    all_agent_pred_frames.append(labeled_frames)
                print("Finish rendering agent preds.")

                agent_gt_frames, _ = render_and_mux(
                    verts_txvx3=all_verts["agent_gt"],
                    faces_fx3=faces,
                    tmp_path=agent_gt_out_tmp_path,
                    image_size=(512, 512),
                    fps=args.fps,
                    device=device,
                    batch_size=args.batch_size,
                    show_tqdm=args.use_tqdm,
                    label="Agent GT",
                )
                print("Finish rendering agent gt.")

                user_gt_frames, user_adj_wav = render_and_mux(
                    verts_txvx3=all_verts["user_gt"],
                    faces_fx3=faces,
                    wav_path=user_wav_path,
                    tmp_path=user_gt_out_tmp,
                    device=device,
                    image_size=(512, 512),
                    fps=args.fps,
                    batch_size=args.batch_size,
                    show_tqdm=args.use_tqdm,
                    label="User",
                )
                print("Finish rendering user gt.")

                assert agent_adj_wav is not None and user_adj_wav is not None
                combined_adj_wav = mix_wavs(
                    [agent_adj_wav, user_adj_wav], tmp_dir=agent_pred_out_tmp_paths[0]
                )
                print("Finish mixing audios.")

                if "sepaker1" in timar_agent_npy_path:
                    if all_verts["agent_neck_mean"][0] > all_verts["user_neck_mean"][0]:
                        combined_frames = concat_frames_horizontally(
                            [
                                *reversed(all_agent_pred_frames),
                                agent_gt_frames,
                                user_gt_frames,
                            ]
                        )
                    else:
                        combined_frames = concat_frames_horizontally(
                            [user_gt_frames, agent_gt_frames * all_agent_pred_frames]
                        )
                else:
                    if all_verts["user_neck_mean"][0] > all_verts["agent_neck_mean"][0]:
                        combined_frames = concat_frames_horizontally(
                            [user_gt_frames, agent_gt_frames, *all_agent_pred_frames]
                        )
                    else:
                        combined_frames = concat_frames_horizontally(
                            [
                                *reversed(all_agent_pred_frames),
                                agent_gt_frames,
                                user_gt_frames,
                            ]
                        )
                print("Finish concat videos.")

                combined_out_mp4 = os.path.join(
                    output_path, name.replace(".npy", "_agent_user.mp4")
                )
                combined_frames_path = os.path.join(
                    output_path, name.replace(".npy", "frames")
                )
                os.makedirs(combined_frames_path, exist_ok=True)
                for k in range(len(combined_frames)):
                    imageio.imwrite(
                        os.path.join(combined_frames_path, f"{k:03d}.png"),
                        combined_frames[k],
                    )
                export_video_with_audio(
                    combined_frames, combined_adj_wav, combined_out_mp4, args.fps
                )

                # create a flag file
                os.mknod(os.path.join(output_path, "render_done.flag"))

                print("Done.")
                succ = True
                # args.batch_size = arg.batch_size * 2
                # print(f"Increase batch size to {args.batch_size} for next sample.")
            except Exception as e:
                args.batch_size = max(1, args.batch_size // 2)
                print(
                    f"[ERROR] Decrease batch size to {args.batch_size} due to exception: {e}. Retrying..."
                )
                succ = False
                raise e


if __name__ == "__main__":
    main()
