import math
from functools import partial
from typing import Dict, Optional, Tuple, Type, cast

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from transformers import Wav2Vec2Model

from .diffloss import DiffLoss


class SpeechEncoder:
    def __init__(self):
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        self.wav2vec.eval()

    def __call__(self, speech_chunk, num_frames):
        if speech_chunk.device != next(self.wav2vec.parameters()).device:
            self.wav2vec = self.wav2vec.to(speech_chunk.device)
        hidden_states = self.wav2vec.feature_extractor(speech_chunk)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = _linear_interpolation(hidden_states, output_len=num_frames)
        hidden_states = self.wav2vec.feature_projection(hidden_states)[0]
        speech_chunk_embs = self.wav2vec.encoder(
            hidden_states,
            output_attentions=False,
        ).last_hidden_state
        return speech_chunk_embs


def _linear_interpolation(features, output_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode="linear"
    )
    return output_features.transpose(1, 2)


def _build_causal_turn_mask(turn_ids: torch.Tensor) -> torch.Tensor:
    """
    Additive attention mask for causal-by-turn attention.
    Inside a turn: bidirectional; across turns: only attend to keys from earlier-or-equal turns.
    Args:
        turn_ids: [B, L] Long turn id per token (L == N_keep for encoder; L == T for decoder)
    Returns:
        attn_mask: [B, 1, L, L] float32 with {0.0, -inf}
    """
    B, L = turn_ids.shape
    device = turn_ids.device
    q_tid = turn_ids.unsqueeze(-1)  # [B, L, 1]
    k_tid = turn_ids.unsqueeze(-2)  # [B, 1, L]
    allowed = k_tid <= q_tid  # [B, L, L] bool
    attn_mask = torch.zeros(B, 1, L, L, device=device, dtype=torch.float32)
    attn_mask.masked_fill_(~allowed.unsqueeze(1), float("-inf"))
    return attn_mask


def _turn_offsets(num_frames: int, num_special_tokens: int):
    """
    Layout inside one turn:
      0: <TURN_START>
      1: <USER_SPEECH_START>, [F frames], <USER_SPEECH_END>
      2: <USER_MOTION_START>, [F frames], <USER_MOTION_END>
      3: <AGENT_SPEECH_START>, [F frames], <AGENT_SPEECH_END>
      4: <AGENT_MOTION_START>, [F frames], <AGENT_MOTION_END>
      5: <TURN_END>
    Returns a dict of slices for frame segments and a list of special positions.
    """
    # We assume num_special_tokens == 10 (as you defined), and the order is:
    # 0: turn_start, 1: u_sp_s, 2: u_sp_e, 3: u_mo_s, 4: u_mo_e,
    # 5: a_sp_s,    6: a_sp_e, 7: a_mo_s, 8: a_mo_e, 9: turn_end
    # Total length per turn: 4*F + 10
    T = 4 * num_frames + num_special_tokens
    cur = 0
    # specials layout indexes inside turn
    pos = {}
    pos["turn_start"] = (cur, cur + 1)
    cur += 1
    pos["u_sp_s"] = (cur, cur + 1)
    cur += 1
    pos["u_sp"] = (cur, cur + num_frames)
    cur += num_frames
    pos["u_sp_e"] = (cur, cur + 1)
    cur += 1
    pos["u_mo_s"] = (cur, cur + 1)
    cur += 1
    pos["u_mo"] = (cur, cur + num_frames)
    cur += num_frames
    pos["u_mo_e"] = (cur, cur + 1)
    cur += 1
    pos["a_sp_s"] = (cur, cur + 1)
    cur += 1
    pos["a_sp"] = (cur, cur + num_frames)
    cur += num_frames
    pos["a_sp_e"] = (cur, cur + 1)
    cur += 1
    pos["a_mo_s"] = (cur, cur + 1)
    cur += 1
    pos["a_mo"] = (cur, cur + num_frames)
    cur += num_frames
    pos["a_mo_e"] = (cur, cur + 1)
    cur += 1
    pos["turn_end"] = (cur, cur + 1)
    cur += 1
    assert cur == T, "offset building error"
    return pos, T


def _build_agent_orders(agent_mask: torch.Tensor):
    """
    Build a fixed permutation over agent-motion indices only.
    Args:
        agent_mask: [B, T] bool (True on agent-motion positions of the *current turn*)
    Returns:
        orders_agent: [B, N_agent] long, permutation of agent indices
        agent_idx:    [B, N_agent] long, sorted agent indices (unused but handy to debug)
    """
    B, T = agent_mask.shape
    device = agent_mask.device
    base = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
    agent_idx = base[agent_mask].view(B, -1)  # [B, N_agent]
    noise = torch.rand_like(agent_idx.float())
    orders_agent = agent_idx.gather(
        1, noise.argsort(dim=1)
    )  # shuffle within agent group
    return orders_agent.long(), agent_idx.long()


def _select_mask_to_pred_cosine(
    agent_mask: torch.Tensor, orders_agent: torch.Tensor, step: int, num_iter: int
) -> torch.Tensor:
    """
    Cosine schedule over *agent-only* positions.
    - target_next = floor(N_agent * cos(pi/2 * (step+1)/num_iter))
      (desired #agent still masked *after this step*)
    - Ensure at least 1 stays masked until the final step.
    - k = max(1, remain - target_next)  (how many to unmask *this step*)
    - Pick the first k currently-masked agent indices in orders_agent order.

    Args:
        agent_mask:   [B, T] bool (currently masked agent positions)
        orders_agent: [B, N_agent] long (perm over agent indices)
    Returns:
        mask_to_pred: [B, T] bool (positions to predict *this step*)
    """
    B, T = agent_mask.shape
    device = agent_mask.device
    N_agent = orders_agent.size(1)

    # Cosine target for "masked count after this step"
    ratio = math.cos(math.pi / 2.0 * (step + 1) / max(1, num_iter))
    target_next = torch.floor(
        torch.full((B,), N_agent * ratio, device=device)
    ).long()  # [B]

    # Current remaining masked
    remain = agent_mask.sum(dim=1)  # [B]

    # Keep at least 1 masked until the last step; on the final step unmask all remaining
    if step < num_iter - 1:
        target_next = torch.minimum(
            torch.maximum(torch.ones_like(target_next), remain - 1), target_next
        )
    else:
        target_next = torch.zeros_like(target_next)

    # Quota to unmask this step: k = remain - target_next  (>=1 except maybe when remain==0)
    k = (remain - target_next).clamp_min(1)  # [B]

    # Choose the first k *currently masked* in orders_agent
    masked_in_order = agent_mask.gather(1, orders_agent)  # [B, N_agent] bool
    csum = masked_in_order.cumsum(dim=1)  # [B, N_agent]
    take = masked_in_order & (csum <= k.unsqueeze(1))  # [B, N_agent]

    mask_to_pred = torch.zeros(B, T, dtype=torch.bool, device=device)
    if take.any():
        row = torch.arange(B, device=device).unsqueeze(1).expand_as(take)
        mask_to_pred[row[take], orders_agent[take]] = True  # scatter to [B,T]
    return mask_to_pred


def _concat_history(
    current_tokens,
    current_mask,
    current_meta,
    history,
    T_max: int,
    mask_history_agent: bool = False,
):
    """
    Concatenate history (if any) with the current turn; then left-crop to last T_max tokens.
    History dict is expected to contain tensors with the same fields as meta and a 'tokens' key.
    """
    B, Tcur, D = current_tokens.shape
    device = current_tokens.device

    # --- Compute absolute turn_id for the current chunk by offsetting with history ---
    if history is None or ("turn_id" not in history):
        # No history: keep current turn_ids as provided by get_one_turn_seq (typically zeros)
        cur_turn_id = current_meta["turn_id"]
    else:
        # Offset = last history turn id + 1, per sample
        last_tid = history["turn_id"][:, -1]  # [B]
        offset = last_tid + 1  # [B]
        cur_turn_id = current_meta["turn_id"] + offset.unsqueeze(1)  # [B, Tcur]
    # Replace in meta (do not modify original dict outside)
    current_meta = dict(current_meta)
    current_meta["turn_id"] = cur_turn_id

    # --- Stitch history + current ---
    if history is None or ("tokens" not in history):
        tokens = current_tokens
        mask = current_mask
        meta = current_meta
    else:
        tokens = torch.cat(
            [history["tokens"], current_tokens], dim=1
        )  # [B, Th+Tcur, D]
        # History part is never generated in this call; only current agent motion is True
        hist_false = torch.zeros_like(history["tokens"][:, :, 0], dtype=torch.bool)
        if mask_history_agent:
            # Mask history agent motion too (for training with history masking)
            hist_agent = history["mask_agent_motion"]
            hist_false = hist_false | hist_agent
        mask = torch.cat([hist_false, current_mask], dim=1)  # [B, Th+Tcur]

        meta = {}
        for k in (
            "token_type",
            "role_id",
            "modality",
            "frame_idx",
            "mask_agent_motion",
            "turn_id",
        ):
            if (history is not None) and (k in history) and (k in current_meta):
                meta[k] = torch.cat(
                    [history[k], current_meta[k]], dim=1
                )  # [B, Th+Tcur]
            else:
                meta[k] = current_meta[k]

    # --- Left-crop to the most recent T_max tokens (tokens/mask/meta in sync) ---
    B, Ttot, D = tokens.shape
    if Ttot > T_max:
        sl = slice(Ttot - T_max, Ttot)
        tokens = tokens[:, sl]
        mask = mask[:, sl]
        for k in meta:
            meta[k] = meta[k][:, sl]

    return tokens, mask, meta


class TIMAR(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    turn_start_token = "<TRUN>"
    user_speech_start_token = "<USER_SPEECH>"
    user_speech_end_token = "<USER_SPEECH/>"
    user_motion_start_token = "<USER_MOTION>"
    user_motion_end_token = "<USER_MOTION/>"
    agent_speech_start_token = "<AGENT_SPEECH>"
    agent_speech_end_token = "<AGENT_SPEECH/>"
    agent_motion_start_token = "<AGENT_MOTION>"
    agent_motion_end_token = "<AGENT_MOTION/>"
    turn_end_token = "<TRUN/>"

    special_tokens = [
        turn_start_token,
        user_speech_start_token,
        user_speech_end_token,
        user_motion_start_token,
        user_motion_end_token,
        agent_speech_start_token,
        agent_speech_end_token,
        agent_motion_start_token,
        agent_motion_end_token,
        turn_end_token,
    ]

    def __init__(
        self,
        motion_fps=25,
        speech_frequency=16_000,
        chunk_second=1,
        max_context_time=8,
        speech_snippet_embed_dim=512,
        motion_embed_dim=56,
        token_embed_dim=512,
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        use_causal_attn=False,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        mask_ratio_min=0.7,
        cond_drop_prob=0.1,
        attn_dropout=0.1,
        proj_dropout=0.1,
        use_diffloss=True,
        diffloss_d=3,
        diffloss_w=1024,
        clip_denoised=False,
        predict_xstart=True,
        noise_schedule="cosine",
        use_kl=False,
        rescale_learned_sigmas=False,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=False,
        use_encoder_only=True,
    ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.motion_fps = motion_fps
        self.speech_frequency = speech_frequency
        self.chunk_second = chunk_second
        self.max_context_time = max_context_time
        self.speech_snippet_embed_dim = speech_snippet_embed_dim
        self.motion_embed_dim = motion_embed_dim
        self.num_special_tokens = len(self.special_tokens)

        self.seq_len = self.max_context_time * (
            2 * 2 * self.motion_fps * self.chunk_second + self.num_special_tokens
        )
        self.token_embed_dim = token_embed_dim
        self.grad_checkpointing = grad_checkpointing
        self.use_encoder_only = use_encoder_only

        self.motion_encoder = nn.Sequential(
            nn.Linear(motion_embed_dim, int(token_embed_dim // 2)),
            nn.ReLU(),
            nn.Linear(int(token_embed_dim // 2), token_embed_dim),
            nn.ReLU(),
        )
        self.speech_encoder = SpeechEncoder()
        self.speech_projection = nn.Linear(speech_snippet_embed_dim, token_embed_dim)

        self.sep_tok_emb = nn.Embedding(self.num_special_tokens, token_embed_dim)
        self.cond_drop_prob = cond_drop_prob
        # Fake class embedding for CFG's unconditional generation (i.e., no user_motion)
        self.fake_latent = nn.Parameter(torch.zeros(1, self.token_embed_dim))

        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        self.mask_token = nn.Parameter(
            torch.zeros(
                1, 1, decoder_embed_dim if not use_encoder_only else encoder_embed_dim
            )
        )

        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len, encoder_embed_dim)
        )
        self.encoder_blocks = nn.ModuleList(
            [
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_dropout,
                    attn_drop=attn_dropout,
                )
                for _ in range(
                    encoder_depth
                    if not use_encoder_only
                    else decoder_depth + encoder_depth
                )
            ]
        )
        self.encoder_norm = norm_layer(encoder_embed_dim)

        if not use_encoder_only:
            # --------------------------------------------------------------------------
            # MAR decoder specifics
            self.decoder_embed = nn.Linear(
                encoder_embed_dim, decoder_embed_dim, bias=True
            )
            self.decoder_pos_embed_learned = nn.Parameter(
                torch.zeros(1, self.seq_len, decoder_embed_dim)
            )

            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                        proj_drop=proj_dropout,
                        attn_drop=attn_dropout,
                    )
                    for _ in range(decoder_depth)
                ]
            )

            self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len,
                decoder_embed_dim if not use_encoder_only else encoder_embed_dim,
            )
        )

        self.use_causal_attn = use_causal_attn

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.use_diffloss = use_diffloss
        hdim = decoder_embed_dim if not use_encoder_only else encoder_embed_dim
        if use_diffloss:
            self.diffloss = DiffLoss(
                target_channels=self.motion_embed_dim,
                z_channels=hdim,
                width=diffloss_w,
                depth=diffloss_d,
                predict_xstart=predict_xstart,
                clip_denoised=clip_denoised,
                num_sampling_steps=num_sampling_steps,
                grad_checkpointing=grad_checkpointing,
                noise_schedule=noise_schedule,
                use_kl=use_kl,
                rescale_learned_sigmas=rescale_learned_sigmas,
            )
            self.diffusion_batch_mul = diffusion_batch_mul
        else:
            self.mlp_head = torch.nn.Sequential(
                torch.nn.Linear(hdim, hdim // 2),
                torch.nn.GELU(),
                torch.nn.Linear(hdim // 2, self.motion_embed_dim),
            )

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.sep_tok_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=0.02)
        if not self.use_encoder_only:
            torch.nn.init.normal_(self.decoder_pos_embed_learned, std=0.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def sample_orders(self, meta: dict) -> torch.Tensor:
        """
        Build per-sample generation orders:
        - All AGENT_MOTION indices come first (shuffled within the group),
        then all the remaining indices (shuffled).
        - This keeps AR semantics (agent-first) and allows masking a prefix later.

        Args:
            meta: dict with "mask_agent_motion": BoolTensor [B, T]

        Returns:
            orders: LongTensor [B, T], a permutation per sample.
        """
        am = meta["mask_agent_motion"]  # [B, T] bool
        B, T = am.shape
        device = am.device

        # Vectorized "group-then-shuffle": agent group score=0, others score=1; add noise to shuffle.
        group = (~am).float()  # agent:0, rest:1
        noise = torch.rand(B, T, device=device)  # [0,1)
        scores = group + noise  # agent in [0,1), rest in [1,2)
        orders = scores.argsort(dim=1)  # ascending -> agent first
        return orders.long()

    def random_masking(
        self, x: torch.Tensor, orders: torch.Tensor, meta: dict
    ) -> torch.Tensor:
        """
        Create MAR diffusion mask for AGENT_MOTION with a controllable ratio.

        Policy:
        - Only AGENT_MOTION tokens are eligible to be masked/generated.
        - The masked fraction is drawn from self.mask_ratio_generator (one rate per batch),
            and applied to the count of AGENT_MOTION tokens in each sample.
        - Speech tokens are never masked here.
        - USER_MOTION is not handled here (it will be mixed with fake_latent in the encoder).

        Args:
            x:      [B, T, D]
            orders: [B, T], permutation with AGENT_MOTION in the front (from sample_orders)
            meta:   dict with "mask_agent_motion": BoolTensor [B, T]

        Returns:
            gen_mask: BoolTensor [B, T], True at positions to be generated by MAR.
        """
        B, T, _ = x.shape
        device = x.device
        am = meta["mask_agent_motion"].to(device)  # [B, T] bool

        # Draw a single ratio for the whole mini-batch.
        mask_rate = float(self.mask_ratio_generator.rvs(1)[0])
        # Number of agent tokens per sample and how many to mask.
        num_agent = am.sum(dim=1)  # [B]
        num_to_mask = torch.ceil(num_agent * mask_rate).to(torch.long)  # [B]

        gen_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        if num_to_mask.max().item() == 0:
            return gen_mask

        # Since orders has all AGENT_MOTION at the front, masking the first K positions per row
        # achieves "mask a subset of agent-motion"; handle variable K across the batch.
        k_max = int(num_to_mask.max().item())
        idx_pref = orders[:, :k_max]  # [B, k_max], agent-only region by construction

        # Build a per-row boolean for how many of the first k_max to keep
        row = torch.arange(B, device=device).unsqueeze(1).expand(B, k_max)  # [B, k_max]
        col = idx_pref
        keep = (
            torch.arange(k_max, device=device).unsqueeze(0).expand(B, k_max)
        )  # [B, k_max]
        keep = keep < num_to_mask.unsqueeze(1)  # [B, k_max] bool

        # Scatter-set True only where "keep" is True
        if keep.any():
            gen_mask[row[keep], col[keep]] = True

        # Safety: ensure no non-agent positions are masked (noop if orders is correct)
        gen_mask &= am
        return gen_mask

    def forward_mae_encoder(self, x, mask, meta):
        """
        Training-time hard replacement on USER tokens (role_id==0) with fake_latent (Bernoulli switch).
        Only non-masked tokens are encoded. Optional causal-by-turn attention.
        Args:
            x:    [B, T, D_in]
            mask: [B, T] bool  (True => AGENT_MOTION to be generated; drop from encoder)
            meta: dict with "role_id"[B,T], "turn_id"[B,T], ...
        Returns:
            enc_out: [B, N_keep, D]
        """
        # Project
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # Hard replacement on USER tokens during training
        if self.training:
            is_user_token = meta["role_id"].eq(0)  # [B, T]
            p = float(getattr(self, "cond_drop_prob", 0.0))
            if p > 0.0 and is_user_token.any():
                m = (torch.rand(B, T, device=device) < p) & is_user_token  # [B, T]
                m = m.unsqueeze(-1).to(dtype)  # [B, T, 1]
                fake = self.fake_latent.to(device=device, dtype=dtype).view(1, 1, -1)
                x = m * fake + (1.0 - m) * x

        x = self.z_proj(x)  # [B, T, D]
        D = x.shape[-1]
        # Positional + pre-norm
        x = x + self.encoder_pos_embed_learned[:, :T, :]
        x = self.z_proj_ln(x)

        # Keep only conditional tokens (drop masked/AGENT_MOTION)
        keep = ~mask  # [B, T]
        if keep.sum() == 0:
            return x.new_zeros(B, 1, D)  # safety

        x_keep = x[keep].view(B, -1, D)  # [B, N_keep, D]
        turn_keep = meta["turn_id"][keep].view(B, -1)  # [B, N_keep]

        # Optional causal-by-turn mask (additive {0,-inf})
        attn_mask = None
        if getattr(self, "use_causal_attn", False):
            attn_mask = _build_causal_turn_mask(turn_keep)  # [B, 1, N_keep, N_keep]

        # Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for block in self.encoder_blocks:
                x_keep = checkpoint(
                    lambda y, m: block(y, attn_mask=m), x_keep, attn_mask
                )
        else:
            for block in self.encoder_blocks:
                x_keep = block(x_keep, attn_mask=attn_mask)
        x_keep = self.encoder_norm(x_keep)
        return x_keep

    def forward_mae_decoder(self, x, mask, meta):
        """
        Reconstruct full-length features, with optional causal-by-turn attention across the full sequence.
        Args:
            x:    [B, N_keep, D_enc]  encoder output (kept tokens only)
            mask: [B, T] bool         True at AGENT_MOTION (these were dropped in encoder)
            meta: dict with "turn_id"[B,T], ...
        Returns:
            out:  [B, T, D_dec]
        """
        B, T = mask.shape

        # Project to decoder dim
        x = self.decoder_embed(x)  # [B, N_keep, D_dec]
        D_dec = x.size(-1)
        device, dtype = x.device, x.dtype

        # Canvas with mask_token then scatter kept features back
        keep = ~mask  # [B, T]
        out = (
            self.mask_token.to(device=device, dtype=dtype)
            .view(1, 1, D_dec)
            .expand(B, T, D_dec)
            .clone()
        )
        out[keep] = x.reshape(-1, D_dec)  # [B, T, D_dec]

        # Decoder positional embedding
        out = out + self.decoder_pos_embed_learned[:, :T, :].to(
            device=device, dtype=dtype
        )

        # Optional causal-by-turn mask on FULL sequence
        attn_mask = None
        if getattr(self, "use_causal_attn", False):
            turn_full = meta["turn_id"][:, :T]  # [B, T]
            attn_mask = _build_causal_turn_mask(turn_full)  # [B, 1, T, T]

        # Transformer decoder
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for block in self.decoder_blocks:
                out = checkpoint(lambda y, m: block(y, attn_mask=m), out, attn_mask)
        else:
            for block in self.decoder_blocks:
                out = block(out, attn_mask=attn_mask)
        out = self.decoder_norm(out)

        # Diffusion positional embedding
        out = out + self.diffusion_pos_embed_learned[:, :T, :].to(
            device=device, dtype=dtype
        )
        return out

    def forward_encoder_only(self, x, mask, meta):
        # Project
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # Hard replacement on USER tokens during training
        if self.training:
            is_user_token = meta["role_id"].eq(0)  # [B, T]
            p = float(getattr(self, "cond_drop_prob", 0.0))
            if p > 0.0 and is_user_token.any():
                m = (torch.rand(B, T, device=device) < p) & is_user_token  # [B, T]
                m = m.unsqueeze(-1).to(dtype)  # [B, T, 1]
                fake = self.fake_latent.to(device=device, dtype=dtype).view(1, 1, -1)
                x = m * fake + (1.0 - m) * x

        x = self.z_proj(x)  # [B, T, D]
        D = x.shape[-1]

        mask_token = self.mask_token.to(device=x.device, dtype=x.dtype)  # [D]
        x = torch.where(mask.unsqueeze(-1), mask_token.view(1, 1, D), x)

        # Positional + pre-norm
        x = x + self.encoder_pos_embed_learned[:, :T, :]
        x = self.z_proj_ln(x)

        # Optional causal-by-turn mask on FULL sequence
        attn_mask = None
        if getattr(self, "use_causal_attn", False):
            turn_full = meta["turn_id"][:, :T]  # [B, T]
            attn_mask = _build_causal_turn_mask(turn_full)  # [B, 1, T, T]

        # Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():  # type: ignore
            for block in self.encoder_blocks:
                x = checkpoint(lambda y, m: block(y, attn_mask=m), x, attn_mask)
        else:
            for block in self.encoder_blocks:
                x = block(x, attn_mask=attn_mask)
        out = self.encoder_norm(x)

        # Diffusion positional embedding
        out = out + self.diffusion_pos_embed_learned[:, :T, :].to(
            device=device, dtype=dtype
        )
        return out

    def forward_diffloss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward_mseloss(
        self,
        z_sel: torch.Tensor,  # [B, n, D_dec]  decoder features at agent indices
        gt_sel: torch.Tensor,  # [B, n, D_mo]   GT motion latents at same indices
        mask_sel: torch.Tensor,  # [B, n]         True where masked (to be supervised)
    ) -> torch.Tensor:
        """
        Compute MLP regression loss only on masked agent-motion positions.
        """
        B, n, _ = z_sel.shape
        if n == 0:
            return z_sel.new_tensor(0.0, requires_grad=True)

        # Predict motion latent from decoder features
        pred = self.mlp_head(z_sel)  # [B, n, D_mo]

        # Select masked positions
        m = mask_sel.bool()
        if not m.any():
            return z_sel.new_tensor(0.0, requires_grad=True)

        pred_m = pred[m]  # [N_masked, D_mo]
        gt_m = gt_sel[m]  # [N_masked, D_mo]

        loss_exp = F.mse_loss(pred_m[:, :50], gt_m[:, :50])
        loss_jaw = F.mse_loss(pred_m[:, 50:53], gt_m[:, 50:53])
        loss_neck = F.mse_loss(pred_m[:, 53:56], gt_m[:, 53:56])
        loss = loss_exp + loss_jaw + loss_neck

        return loss

    def forward_loss(self, z, gt_motions, mask):
        if self.use_diffloss:
            return self.forward_diffloss(z, gt_motions, mask)
        else:
            return self.forward_mseloss(z, gt_motions, mask)

    def forward(self, speech_1, motion_1, speech_2, motion_2):
        # get input sequences
        x, meta, gt_motions = self.build_input_sequences(
            speech_1, motion_1, speech_2, motion_2
        )

        orders = self.sample_orders(meta)
        mask = self.random_masking(x, orders, meta)

        if self.use_encoder_only:
            # encoder only
            z = self.forward_encoder_only(x, mask, meta)
        else:
            # mae encoder
            x = self.forward_mae_encoder(x, mask, meta)
            # mae decoder
            z = self.forward_mae_decoder(x, mask, meta)

        # diffloss
        B, T, D = z.shape
        sel = meta["mask_agent_motion"]  # [B,T] bool
        n = int(sel[0].sum().item())
        idx_t = torch.arange(T, device=z.device).expand(B, T)[sel].view(B, n)  # [B,n]
        z_sel = z.gather(1, idx_t.unsqueeze(-1).expand(-1, -1, D))  # [B,n,D]
        mask_sel = mask.gather(1, idx_t)

        loss = self.forward_loss(z=z_sel, gt_motions=gt_motions, mask=mask_sel)

        return loss

    def get_one_turn_seq(
        self,
        speech_1_chunk: torch.Tensor,  # [B, F, D]
        motion_1_chunk: torch.Tensor,  # [B, F, D]
        speech_2_chunk: torch.Tensor,  # [B, F, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Build one-turn interleaved sequence with specials.
        Returns:
        tokens: [B, 4F + num_special_tokens, D]
        mask:   [B, T] bool, True on AGENT_MOTION (to be generated)
        meta:   dict with token_type, role_id, modality, frame_idx, mask_agent_motion
        """
        B, F, D = speech_1_chunk.shape
        device = speech_1_chunk.device

        # Special token embeddings [num_special_tokens, D]
        tok_ids = torch.arange(self.num_special_tokens, device=device, dtype=torch.long)
        spec = self.sep_tok_emb(tok_ids)  # order must match your defined special tokens

        # Build layout
        pos, T = _turn_offsets(F, self.num_special_tokens)

        # Allocate sequence and meta
        tokens = speech_1_chunk.new_empty(B, T, D)

        token_type = torch.zeros(B, T, dtype=torch.long, device=device)  # 0:special
        role_id = torch.full((B, T), 2, dtype=torch.long, device=device)
        modality = torch.full((B, T), 2, dtype=torch.long, device=device)
        frame_idx = torch.full((B, T), -1, dtype=torch.long, device=device)

        # Fill specials (order must align with self.special_tokens)
        specials_map = [
            ("turn_start", 0),
            ("u_sp_s", 1),
            ("u_sp_e", 2),
            ("u_mo_s", 3),
            ("u_mo_e", 4),
            ("a_sp_s", 5),
            ("a_sp_e", 6),
            ("a_mo_s", 7),
            ("a_mo_e", 8),
            ("turn_end", 9),
        ]
        for name, sid in specials_map:
            a, b = pos[name]
            tokens[:, a:b] = spec[sid].view(1, 1, D).expand(B, 1, D)

        # Fill frames
        arange_F = torch.arange(F, device=device).unsqueeze(0).expand(B, F)

        a, b = pos["u_sp"]
        tokens[:, a:b] = speech_1_chunk
        token_type[:, a:b] = 1
        role_id[:, a:b] = 0
        modality[:, a:b] = 0
        frame_idx[:, a:b] = arange_F

        a, b = pos["u_mo"]
        tokens[:, a:b] = motion_1_chunk
        token_type[:, a:b] = 2
        role_id[:, a:b] = 0
        modality[:, a:b] = 1
        frame_idx[:, a:b] = arange_F

        a, b = pos["a_sp"]
        tokens[:, a:b] = speech_2_chunk
        token_type[:, a:b] = 3
        role_id[:, a:b] = 1
        modality[:, a:b] = 0
        frame_idx[:, a:b] = arange_F

        a, b = pos["a_mo"]
        # NOTE: agent motion tokens will be generated -> put a placeholder (zeros) now.
        tokens[:, a:b] = 0.0
        token_type[:, a:b] = 4
        role_id[:, a:b] = 1
        modality[:, a:b] = 1
        frame_idx[:, a:b] = arange_F

        # Agent-motion mask (to be generated)
        mask_agent_motion = torch.zeros(B, T, dtype=torch.bool, device=device)
        a, b = pos["a_mo"]
        mask_agent_motion[:, a:b] = True

        turn_id = torch.zeros(B, T, dtype=torch.long, device=device)

        meta = {
            "token_type": token_type,
            "role_id": role_id,
            "modality": modality,
            "frame_idx": frame_idx,
            "mask_agent_motion": mask_agent_motion,
            "turn_id": turn_id,
        }
        return tokens, mask_agent_motion, meta

    def sample(
        self,
        speech_1_chunk: torch.Tensor,
        motion_1_chunk: torch.Tensor,
        speech_2_chunk: torch.Tensor,
        history: Optional[dict] = None,
        max_context_time: Optional[int] = None,
        mask_history_agent: bool = False,
        num_iter: int = 64,
        cfg: float = 1.0,
        cfg_schedule: str = "linear",
        temperature: float = 1.0,
        use_random_orders: bool = True,
        progress: bool = False,
    ):
        """
        MAR sampling with history stitching and 8s cropping.
        - If `history` is None: sample a single turn.
        - Else: concatenate history (as conditions) before the current turn; only current agent motion is masked.
        - If concatenated sequence exceeds 8s window, keep the most recent window.
        """
        # Ensure batch dims
        if speech_1_chunk.dim() == 1:
            speech_1_chunk = speech_1_chunk.unsqueeze(0)
        if speech_2_chunk.dim() == 1:
            speech_2_chunk = speech_2_chunk.unsqueeze(0)
        if motion_1_chunk.dim() == 2:
            motion_1_chunk = motion_1_chunk.unsqueeze(0)

        # Check durations
        num_frames = int(self.chunk_second * self.motion_fps)
        s1_dur = speech_1_chunk.size(1) / self.speech_frequency
        s2_dur = speech_2_chunk.size(1) / self.speech_frequency
        m1_dur = motion_1_chunk.size(1) / self.motion_fps
        assert s1_dur == m1_dur == self.chunk_second
        assert s2_dur == self.chunk_second

        # Encode per-frame features
        s1 = self.encode_speech_chunk(
            speech_1_chunk, num_chunks=1, num_frames=num_frames
        ).squeeze(
            1
        )  # [B,F,D]
        m1 = self.encode_motion_chunk(
            motion_1_chunk, num_chunks=1, num_frames=num_frames
        ).squeeze(
            1
        )  # [B,F,D]
        s2 = self.encode_speech_chunk(
            speech_2_chunk, num_chunks=1, num_frames=num_frames
        ).squeeze(
            1
        )  # [B,F,D]

        # Build current turn
        cur_tokens, cur_mask, cur_meta = self.get_one_turn_seq(
            s1, m1, s2
        )  # mask: agent motion only

        # History stitching and max_context_time cropping
        if max_context_time is not None:
            max_context_time = min(max_context_time, self.max_context_time)
        else:
            max_context_time = self.max_context_time
        max_turns = max_context_time // self.chunk_second
        per_turn_T = 4 * num_frames + self.num_special_tokens
        T_max = max_turns * per_turn_T
        tokens, mask, meta = _concat_history(
            cur_tokens, cur_mask, cur_meta, history, T_max, mask_history_agent
        )

        B, T, D = tokens.shape
        tokens_cur = tokens.clone()
        motion_2 = motion_1_chunk.new_empty(B, T, self.motion_embed_dim)
        base_mask = mask.clone()  # only the current turn's agent motion region is True
        # Build agent-only orders once per call
        agent_mask = base_mask  # base_mask == mask_agent_motion over current turn
        orders_agent, agent_idx = _build_agent_orders(
            agent_mask
        )  # [B, N_agent], [B, N_agent]

        if not use_random_orders:
            orders_agent = agent_idx  # always left-to-right
        N_agent = orders_agent.size(1)

        iters = tqdm(range(num_iter)) if progress else range(num_iter)
        for step in iters:
            # ---------------- CFG (unchanged): unconditional branch hard-replaces CURRENT-TURN USER_MOTION with fake_latent ----------------
            x_tokens = tokens_cur
            x_mask = base_mask  # same agent-only mask for both branches

            if cfg != 1.0 and self.use_diffloss:
                B, T, D = tokens_cur.shape
                cur_slice = slice(T - per_turn_T, T)
                is_u_cur = meta["role_id"][:, cur_slice].eq(0)  # [B, per_turn_T]
                fake = self.fake_latent.to(tokens_cur.device, tokens_cur.dtype).view(
                    1, 1, -1
                )
                uncond_tokens = tokens_cur.clone()
                um_view = uncond_tokens[:, cur_slice]
                um_view[is_u_cur] = fake.expand(B, per_turn_T, D)[is_u_cur]
                x_tokens = torch.cat([x_tokens, uncond_tokens], dim=0)  # [2B, T, D]
                x_mask = torch.cat([x_mask, x_mask], dim=0)  # [2B, T]
                meta["turn_id"] = torch.cat([meta["turn_id"], meta["turn_id"]], dim=0)

            if self.use_encoder_only:
                # ---------------- Encoder-only forward ----------------
                z = self.forward_encoder_only(
                    x_tokens, x_mask, meta
                )  # [B or 2B, T, D_z]
            else:
                # ---------------- Encoder / Decoder ----------------
                enc = self.forward_mae_encoder(x_tokens, x_mask, meta)
                z = self.forward_mae_decoder(enc, x_mask, meta)  # [B or 2B, T, D_z]

            # ---------------- Agent-only cosine selection (guaranteed non-empty) ----------------
            mask_to_pred = _select_mask_to_pred_cosine(
                agent_mask, orders_agent, step, num_iter
            )  # [B, T]
            if cfg != 1.0 and self.use_diffloss:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # Safety guard (should not trigger)
            if not mask_to_pred.any():
                break

            # Gather features to predict
            sel = mask_to_pred.nonzero(as_tuple=True)
            z_sel = z[sel]  # [N_sel, D_z]

            # ---------------- CFG schedule driven by cosine progress over agent-only ----------------
            if self.use_diffloss:
                if cfg_schedule == "linear":
                    # After this step, expected masked count is target_next (from selector). Recompute to get a scalar progress.
                    remain = agent_mask.sum(dim=1).float()  # [B]
                    ratio = math.cos(math.pi / 2.0 * (step + 1) / max(1, num_iter))
                    target_next = torch.floor(
                        torch.full((B,), N_agent * ratio, device=z.device)
                    ).float()
                    if step < num_iter - 1:
                        target_next = torch.minimum(
                            torch.maximum(torch.ones_like(target_next), remain - 1.0),
                            target_next,
                        )
                    else:
                        target_next = torch.zeros_like(target_next)
                    unmasked_frac = (N_agent - target_next).clamp_min(0.0) / max(
                        N_agent, 1
                    )
                    cfg_iter = 1.0 + (cfg - 1.0) * float(unmasked_frac.mean().item())
                elif cfg_schedule == "constant":
                    cfg_iter = float(cfg)
                else:
                    raise NotImplementedError

                # ---------------- Diffusion sampling ----------------
                sampled = self.diffloss.sample(
                    z_sel, temperature=temperature, cfg=cfg_iter
                )
                if cfg != 1.0:
                    sampled, _ = sampled.chunk(2, dim=0)
                    mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                    meta["turn_id"] = meta["turn_id"][:B]
            else:
                sampled = self.mlp_head(z_sel)  # [N_sel, D_mo]

            # ---------------- Scatter back & update agent_mask ----------------
            flat = tokens_cur.view(B * T, D)
            motion_2_flat = motion_2.view(B * T, self.motion_embed_dim)
            sel_cond = mask_to_pred[:B].nonzero(as_tuple=True)

            flat[sel_cond[0] * T + sel_cond[1]] = self.motion_encoder(sampled)
            motion_2_flat[sel_cond[0] * T + sel_cond[1]] = sampled

            tokens_cur = flat.view(B, T, D)
            motion_2 = motion_2_flat.view(B, T, self.motion_embed_dim)

            # Mark these positions as unmasked for the next iteration
            agent_mask = agent_mask & (~mask_to_pred[:B])

        # Prepare updated history (keep the stitched-and-filled window as next call's history)
        new_history = {
            "tokens": tokens_cur.detach(),  # [B, T, D]
            "token_type": meta["token_type"].detach(),  # [B, T]
            "role_id": meta["role_id"].detach(),
            "turn_id": meta["turn_id"].detach(),
            "modality": meta["modality"].detach(),
            "frame_idx": meta["frame_idx"].detach(),
            "mask_agent_motion": meta["mask_agent_motion"].detach(),
        }

        # Return the filled current turn slice (agent motion written) + updated history
        return (
            motion_2[:, -(self.chunk_second * self.motion_fps + 2) : -2, :],
            new_history,
        )

    def encode_speech_chunk(
        self, speech: torch.Tensor, num_chunks: int, num_frames: int
    ):
        # speech: [B, T*160000]
        bsz = speech.size(0)
        speech = speech.view(bsz, num_chunks, -1).view(bsz * num_chunks, -1)
        with torch.no_grad():
            speech_chunk_embs = self.speech_encoder(speech, num_frames)
        speech_chunk_tokens = self.speech_projection(speech_chunk_embs)
        speech_chunk_tokens = speech_chunk_tokens.view(bsz, num_chunks, num_frames, -1)
        return speech_chunk_tokens

    def encode_motion_chunk(self, motion, num_chunks, num_frames):
        motion = self.motion_encoder(motion)
        return motion.view(motion.size(0), num_chunks, num_frames, -1)

    def interleave(
        self,
        speech_1: torch.Tensor,
        motion_1: torch.Tensor,
        speech_2: torch.Tensor,
        motion_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Build an interleaved sequence:
        <TURN>
        <USER_SPEECH>  s1[0..F-1]  <USER_SPEECH/>
        <USER_MOTION>  m1[0..F-1]  <USER_MOTION/>
        <AGENT_SPEECH> s2[0..F-1]  <AGENT_SPEECH/>
        <AGENT_MOTION> m2[0..F-1]  <AGENT_MOTION/>
        <TURN/>

        Args:
            speech_1, motion_1, speech_2, motion_2: [B, C, F, D]

        Returns:
            seq:  [B, C*(10+4F), D]
            meta: dict of useful masks/indices (all [B, T] unless noted)
                - token_type: int labels {0:special,1:u_sp,2:u_mo,3:a_sp,4:a_mo}
                - role_id:   {0:user,1:agent,2:special}
                - modality:  {0:speech,1:motion,2:special}
                - turn_id:   [B, T]
                - frame_idx: [-1 for specials, else 0..F-1]
                - mask_agent_motion: bool mask for loss/AR training
                - attn_mask: all-ones (no padding here)
        """
        assert speech_1.dim() == motion_1.dim() == speech_2.dim() == motion_2.dim() == 4
        B, C, F, D = speech_1.shape
        device, dtype = speech_1.device, speech_1.dtype

        # [10, D] special token embeddings
        tok_ids = torch.as_tensor(
            list(range(self.num_special_tokens)), device=device, dtype=torch.long
        )
        spec = self.sep_tok_emb(tok_ids)

        # Per-turn layout: 10 specials + 4 frame blocks
        per_turn_len = self.num_special_tokens + 4 * F
        T = C * per_turn_len

        # Allocate outputs
        seq = speech_1.new_empty(B, T, D)
        token_type = torch.zeros(
            B, T, dtype=torch.long, device=device
        )  # default special
        role_id = torch.full(
            (B, T), 2, dtype=torch.long, device=device
        )  # 2 for specials
        modality = torch.full((B, T), 2, dtype=torch.long, device=device)
        turn_id = torch.empty(B, T, dtype=torch.long, device=device)
        frame_idx = torch.full((B, T), -1, dtype=torch.long, device=device)
        attn_mask = torch.ones(B, T, dtype=torch.bool, device=device)

        # Precompute frame index row once
        arange_F = torch.arange(F, device=device).unsqueeze(0).expand(B, F)

        # Relative offsets inside a single turn
        # 0:TS, 1:USS, 2:Usp[F], 3:USE, 4:UMS, 5:Umo[F], 6:UME,
        # 7:ASS, 8:Asp[F], 9:ASE, 10:AMS, 11:Amo[F], 12:AME, 13:TE
        offs = []
        cur = 0
        for span in [1, 1, F, 1, 1, F, 1, 1, F, 1, 1, F, 1, 1]:
            offs.append((cur, cur + span))
            cur += span
        assert cur == per_turn_len

        # Fill turn by turn
        for c in range(C):
            base = c * per_turn_len
            # Short aliases for this turn [B, F, D]
            s1c, m1c = speech_1[:, c], motion_1[:, c]
            s2c, m2c = speech_2[:, c], motion_2[:, c]

            # Specials (10 of them): expand once per slice to avoid broadcasting overhead
            spec_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # indices into `spec`
            # Map to the 10 positions in our layout
            spec_spans = [offs[i] for i in [0, 1, 3, 4, 6, 7, 9, 10, 12, 13]]
            for tok_i, (a, b) in zip(spec_ids, spec_spans):
                sl = slice(base + a, base + b)
                seq[:, sl] = spec[tok_i].expand(B, 1, D)

            # User speech frames
            a, b = offs[2]
            sl = slice(base + a, base + b)
            seq[:, sl] = s1c
            token_type[:, sl] = 1
            role_id[:, sl] = 0
            modality[:, sl] = 0
            frame_idx[:, sl] = arange_F

            # User motion frames
            a, b = offs[5]
            sl = slice(base + a, base + b)
            seq[:, sl] = m1c
            token_type[:, sl] = 2
            role_id[:, sl] = 0
            modality[:, sl] = 1
            frame_idx[:, sl] = arange_F

            # Agent speech frames
            a, b = offs[8]
            sl = slice(base + a, base + b)
            seq[:, sl] = s2c
            token_type[:, sl] = 3
            role_id[:, sl] = 1
            modality[:, sl] = 0
            frame_idx[:, sl] = arange_F

            # Agent motion frames
            a, b = offs[11]
            sl = slice(base + a, base + b)
            seq[:, sl] = m2c
            token_type[:, sl] = 4
            role_id[:, sl] = 1
            modality[:, sl] = 1
            frame_idx[:, sl] = arange_F

            # Turn id for the whole span
            turn_id[:, base : base + per_turn_len] = c

        # Boolean mask for agent motion (useful for loss / AR target selection)
        mask_agent_motion = token_type.eq(4)

        meta = {
            "token_type": token_type,  # {0:special,1:user_speech,2:user_motion,3:agent_speech,4:agent_motion}
            "role_id": role_id,  # {0:user,1:agent,2:special}
            "modality": modality,  # {0:speech,1:motion,2:special}
            "turn_id": turn_id,  # [0..C-1]
            "frame_idx": frame_idx,  # -1 for specials, else [0..F-1]
            "mask_agent_motion": mask_agent_motion,
            "attn_mask": attn_mask,
        }
        return seq, meta

    def build_input_sequences(self, speech_1, motion_1, speech_2, motion_2):
        # speech: [B, T*160000], motion: [B, T*frequency, 56]

        duration = speech_1.size(1) // self.speech_frequency
        num_chunks = duration // self.chunk_second
        assert (
            duration == num_chunks * self.chunk_second
        ), "Duration should be multiple of chunk_second"
        num_frames = int(self.chunk_second * self.motion_fps)

        speech_1_chunks = self.encode_speech_chunk(
            speech_1, num_chunks=num_chunks, num_frames=num_frames
        )
        motion_1_chunks = self.encode_motion_chunk(
            motion_1, num_chunks=num_chunks, num_frames=num_frames
        )
        speech_2_chunks = self.encode_speech_chunk(
            speech_2, num_chunks=num_chunks, num_frames=num_frames
        )
        motion_2_chunks = self.encode_motion_chunk(
            motion_2, num_chunks=num_chunks, num_frames=num_frames
        )

        x, meta = self.interleave(
            speech_1_chunks, motion_1_chunks, speech_2_chunks, motion_2_chunks
        )

        return x, meta, motion_2.clone().detach()


def timar_base(**kwargs):
    model = TIMAR(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=768,
        decoder_depth=12,
        decoder_num_heads=12,
        mlp_ratio=4,
        norm_layer=cast(Type[nn.LayerNorm], partial(nn.LayerNorm, eps=1e-6)),
        **kwargs,
    )
    return model


def timar_large(**kwargs):
    model = TIMAR(
        encoder_embed_dim=1024,
        encoder_depth=16,
        encoder_num_heads=16,
        decoder_embed_dim=1024,
        decoder_depth=16,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=cast(Type[nn.LayerNorm], partial(nn.LayerNorm, eps=1e-6)),
        **kwargs,
    )
    return model


def timar_huge(**kwargs):
    model = TIMAR(
        encoder_embed_dim=1280,
        encoder_depth=20,
        encoder_num_heads=16,
        decoder_embed_dim=1280,
        decoder_depth=20,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=cast(Type[nn.LayerNorm], partial(nn.LayerNorm, eps=1e-6)),
        **kwargs,
    )
    return model
