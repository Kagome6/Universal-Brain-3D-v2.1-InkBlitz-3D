# -*- coding: utf-8 -*-
"""
brain3d_torch_models.py — Top-Level PyTorch Whole-Brain Neural Architecture Suite (13 Models)
=============================================================================================
Barrier 1 Isolation Rule:
  This module imports `torch` at top level so all 12 `nn.Module` classes live cleanly at
  module scope (zero nested-function indentation) with full IDE/type-checker & pickle support.
  It MUST ONLY be imported inside `GPUInferenceProcess` (`WholeBrainNeuralEngineWrapper.__init__`).

License: GNU General Public License v3.0 (GPL-3.0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from task_profiles_3d import *
from brain3d_config_ipc import *


class TTLinear(nn.Module):
    """2-Core Tensor-Train Factorized Linear Projection (`(i0*i1) -> (o0*o1)` with rank `r`)."""

    def __init__(self, in_p: Tuple[int, int], out_p: Tuple[int, int], rank: int = 16):
        super().__init__()
        self.i0, self.i1 = int(in_p[0]), int(in_p[1])
        self.o0, self.o1 = int(out_p[0]), int(out_p[1])
        self.rank = int(rank)
        scale0 = 1.0 / math.sqrt(self.i0 * self.rank)
        scale1 = 1.0 / math.sqrt(self.i1 * self.rank)
        self.core0 = nn.Parameter(torch.randn(self.i0, self.o0, self.rank) * scale0)
        self.core1 = nn.Parameter(torch.randn(self.rank, self.i1, self.o1) * scale1)
        self.bias = nn.Parameter(torch.zeros(self.o0 * self.o1))

    def forward(self, x):
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, self.i0, self.i1)
        h = torch.einsum("bij,ior->bojr", x2, self.core0)
        y = torch.einsum("bojr,rjp->bop", h, self.core1).reshape(*orig_shape, self.o0 * self.o1)
        return y + self.bias

class TensorTrainFovealEncoder3D(nn.Module):
    """Encodes `16x16x4` dorsal fovea + `16D` proprio + `24D` compass into `32D` latent `z`."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FOVEA_C, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )  # -> (B, 32, 4, 4) = 512
        self.tt_vis = TTLinear((16, 32), (8, 8), rank=16)  # 512 -> 64
        self.ctx_proj = nn.Linear(PROPRIO_DIM + MEM_COMPASS_DIM, 48)
        self.fuse = nn.Sequential(
            nn.Linear(64 + 48, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, LATENT_Z_DIM),
        )
        self.recon_head = nn.Sequential(
            nn.Linear(LATENT_Z_DIM, 64),
            nn.GELU(),
            nn.Linear(64, FOVEA_H * FOVEA_W),
            nn.Sigmoid(),
        )

    def forward(self, fovea_bhwc, proprio, mem_compass):
        x = fovea_bhwc.permute(0, 3, 1, 2).contiguous()
        h_v = self.tt_vis(self.conv(x).flatten(1))
        h_c = F.gelu(self.ctx_proj(torch.cat([proprio, mem_compass], dim=-1)))
        z = torch.tanh(self.fuse(torch.cat([h_v, h_c], dim=-1)))
        return z

    def decode_depth(self, z):
        return self.recon_head(z).reshape(-1, FOVEA_H, FOVEA_W)

class HazardPredictor3D(nn.Module):
    """Fast Amygdala-like immediate threat & crossfire estimator."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_Z_DIM + PROPRIO_DIM + MEM_COMPASS_DIM, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, proprio, mem_compass):
        return self.net(torch.cat([z, proprio, mem_compass], dim=-1)).squeeze(-1)

class TensorTrainD3QNCandidateNet3D(nn.Module):
    """
    Upgraded Dueling Double-DQN Candidate Evaluator (`TensorTrainD3QNCandidateNet3D` v2):
    Scores `M = 8` decoupled `(move_xyz, aim_yp, verb)` candidates with `32D` compass-alignment features
    and outputs both `q_values (B, 8)` and continuous `speed_mod (B, 8)`.
    """

    def __init__(self):
        super().__init__()
        # State trunk: z(32) + gwt(128) + rssm_h(256) + mec(256) + hd(32) + proprio(16) + compass(24) = 744 -> pad 768 (24*32)
        self.state_in_dim = LATENT_Z_DIM + GWT_DIM + RSSM_DETER_DIM + MEC_GRID_DIM + HD_RING_DIM + PROPRIO_DIM + MEM_COMPASS_DIM
        self.tt_state = TTLinear((24, 32), (12, 16), rank=16)  # 768 -> 192
        self.state_norm = nn.LayerNorm(192)
        self.val_head = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1))

        self.cand_enc = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(192 + 64, 96),
            nn.GELU(),
            nn.Linear(96, 1),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(192 + 64, 48),
            nn.GELU(),
            nn.Linear(48, 1),
            nn.Sigmoid(),
        )

    def forward(self, z, gwt_w, rssm_h, mec_code, hd_code, proprio, mem_compass, cand_feats32):
        bsz = z.shape[0]
        raw_s = torch.cat([z, gwt_w, rssm_h, mec_code, hd_code, proprio, mem_compass], dim=-1)
        if raw_s.shape[-1] < 768:
            pad = torch.zeros((bsz, 768 - raw_s.shape[-1]), dtype=raw_s.dtype, device=raw_s.device)
            raw_s = torch.cat([raw_s, pad], dim=-1)
        s_emb = F.gelu(self.state_norm(self.tt_state(raw_s)))  # (B, 192)
        v_s = self.val_head(s_emb)  # (B, 1)

        c_emb = self.cand_enc(cand_feats32)  # (B, 8, 64)
        s_exp = s_emb.unsqueeze(1).expand(-1, NUM_CANDIDATES, -1)
        pair = torch.cat([s_exp, c_emb], dim=-1)  # (B, 8, 256)
        adv = self.adv_head(pair).squeeze(-1)  # (B, 8)
        q_vals = v_s + (adv - adv.mean(dim=-1, keepdim=True))
        spd_mod = 0.25 + 0.75 * self.speed_head(pair).squeeze(-1)
        return q_vals, spd_mod

class CategoricalRSSM3D(nn.Module):
    """
    Upgraded 3-Head Multi-Step Causal Categorical World Model (`CategoricalRSSM3D` v2.1):
      - 1% Unimix categorical latents (`8x8 = 64D`) + recurrent `GRUCell` dynamics
      - Multi-step open-loop GRU imagination (`horizon=3`) across all `M=8` candidates:
          1. `pred_rew (B, 8)`  : Discounted multi-step tactical reward (`0.70/0.20/0.10`)
          2. `pred_surv (B, 8)` : Pessimistic multi-step survival probability (`0.75*s0 + 0.25*min(s1,s2)`)
          3. `pred_adv (B, 8)`  : Multi-step territorial turf / objective advance probability
    """

    def __init__(self):
        super().__init__()
        self.obs_proj = nn.Sequential(
            nn.Linear(LATENT_Z_DIM + MULTIMODAL_EMBED_DIM + PROPRIO_DIM + MEM_COMPASS_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.act_proj = nn.Sequential(nn.Linear(32, 64), nn.GELU())
        self.gru = nn.GRUCell(RSSM_STOCH_FLAT + 64, RSSM_DETER_DIM)
        self.prior_net = nn.Linear(RSSM_DETER_DIM, RSSM_STOCH_FLAT)
        self.post_net = nn.Linear(RSSM_DETER_DIM + 128, RSSM_STOCH_FLAT)

        self.imag_step = nn.Sequential(
            nn.Linear(RSSM_DETER_DIM + RSSM_STOCH_FLAT + 64, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        self.rew_head = nn.Linear(192, 1)
        self.surv_head = nn.Linear(192, 1)
        self.adv_head = nn.Linear(192, 1)

    def _st_categorical(self, logits):
        bsz = logits.shape[0]
        lg = logits.reshape(bsz, RSSM_STOCH_CATS, RSSM_STOCH_CLASSES)
        probs = 0.99 * F.softmax(lg, dim=-1) + (0.01 / float(RSSM_STOCH_CLASSES))
        idx = torch.argmax(probs, dim=-1)
        one_hot = F.one_hot(idx, RSSM_STOCH_CLASSES).to(probs.dtype)
        st = one_hot + probs - probs.detach()
        return st.reshape(bsz, RSSM_STOCH_FLAT), probs

    def observe_step(self, prev_h, prev_stoch, prev_cand_feat32, z, slot_ctx64, proprio, mem_compass):
        act_e = self.act_proj(prev_cand_feat32)
        h_next = self.gru(torch.cat([prev_stoch, act_e], dim=-1), prev_h)
        prior_logits_flat = self.prior_net(h_next)
        obs_e = self.obs_proj(torch.cat([z, slot_ctx64, proprio, mem_compass], dim=-1))
        post_logits_flat = self.post_net(torch.cat([h_next, obs_e], dim=-1))
        post_stoch, post_probs = self._st_categorical(post_logits_flat)
        _, prior_probs = self._st_categorical(prior_logits_flat)
        return h_next, post_stoch, post_probs, prior_probs

    def imagine_step_open_loop(self, prev_h, prev_stoch, trans_cand_feat32, eval_cand_feat32=None):
        act_trans = self.act_proj(trans_cand_feat32)
        h_next = self.gru(torch.cat([prev_stoch, act_trans], dim=-1), prev_h)
        prior_stoch, prior_probs = self._st_categorical(self.prior_net(h_next))
        act_eval = self.act_proj(eval_cand_feat32) if eval_cand_feat32 is not None else act_trans
        im_h = self.imag_step(torch.cat([h_next, prior_stoch, act_eval], dim=-1))
        r_step = self.rew_head(im_h).squeeze(-1)
        s_step = torch.sigmoid(self.surv_head(im_h).squeeze(-1))
        a_step = torch.sigmoid(self.adv_head(im_h).squeeze(-1))
        return h_next, prior_stoch, r_step, s_step, a_step, prior_probs

    def imagine_candidates(self, h_t, stoch_t, cand_feats32, horizon: int = 3):
        bsz = h_t.shape[0]
        act_e = self.act_proj(cand_feats32)  # (B, 8, 64)
        base0 = torch.cat([h_t, stoch_t], dim=-1).unsqueeze(1).expand(-1, NUM_CANDIDATES, -1)
        im_h0 = self.imag_step(torch.cat([base0, act_e], dim=-1))  # (B, 8, 192)
        r0 = self.rew_head(im_h0).squeeze(-1)
        s0 = torch.sigmoid(self.surv_head(im_h0).squeeze(-1))
        a0 = torch.sigmoid(self.adv_head(im_h0).squeeze(-1))
        if int(horizon) <= 1:
            return r0, s0, a0

        # Multi-step open-loop recurrent GRU + prior rollout across all B * 8 candidate branches
        act_flat = act_e.reshape(bsz * NUM_CANDIDATES, 64)
        h_cur = h_t.unsqueeze(1).expand(-1, NUM_CANDIDATES, -1).reshape(bsz * NUM_CANDIDATES, RSSM_DETER_DIM)
        st_cur = stoch_t.unsqueeze(1).expand(-1, NUM_CANDIDATES, -1).reshape(bsz * NUM_CANDIDATES, RSSM_STOCH_FLAT)

        r_list = [r0]
        s_list = [s0]
        a_list = [a0]
        for _ in range(int(horizon) - 1):
            h_cur = self.gru(torch.cat([st_cur, act_flat], dim=-1), h_cur)
            st_cur, _ = self._st_categorical(self.prior_net(h_cur))
            im_hk = self.imag_step(torch.cat([h_cur, st_cur, act_flat], dim=-1))
            r_list.append(self.rew_head(im_hk).view(bsz, NUM_CANDIDATES))
            s_list.append(torch.sigmoid(self.surv_head(im_hk).view(bsz, NUM_CANDIDATES)))
            a_list.append(torch.sigmoid(self.adv_head(im_hk).view(bsz, NUM_CANDIDATES)))

        if len(r_list) >= 3:
            p_rew = 0.70 * r_list[0] + 0.20 * r_list[1] + 0.10 * r_list[2]
            p_surv = 0.75 * s_list[0] + 0.25 * torch.minimum(s_list[1], s_list[2])
            p_adv = 0.70 * a_list[0] + 0.20 * a_list[1] + 0.10 * a_list[2]
        else:
            p_rew = 0.75 * r_list[0] + 0.25 * r_list[1]
            p_surv = 0.75 * s_list[0] + 0.25 * s_list[1]
            p_adv = 0.75 * a_list[0] + 0.25 * a_list[1]
        return p_rew, p_surv, p_adv

class ObjectSlotPredictiveCortex3D(nn.Module):
    """
    Asymmetric 4-Slot Object-Centric Predictive Visual Cortex (`ObjectSlotPredictiveCortex3D` v2.1):
      - Slot 0 captures wide background / floor surface field; Slots 1..3 bind to salient foreground objects
      - Closes the recurrent Predictive Coding & Top-Down Precision Spotlight loop (`pred_err_16` + `topdown_gain_16`)
      - Injects zero-gated foreground residual `delta_z = tanh(z_res_gate) * out_z_proj(fg_pool)` into `latent_z`
      - Predicts action- & RSSM-conditioned next `16x16x8` visual field (`predict_next_visual_field`)
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(VIS8_C, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, OBJECT_SLOT_DIM, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )  # -> (B, 32, 8, 8)
        self.err_spot_conv = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.err_spot_conv.weight, gain=0.20)
        nn.init.zeros_(self.err_spot_conv.bias)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 8),
            torch.linspace(-1.0, 1.0, 8),
            indexing="ij",
        )
        self.register_buffer("grid_xy", torch.stack([xx, yy], dim=-1).reshape(1, 64, 2))
        u16 = torch.linspace(-1.0, 1.0, FOVEA_W)
        v16 = torch.linspace(-1.0, 1.0, FOVEA_H)
        vv16, uu16 = torch.meshgrid(v16, u16, indexing="ij")
        uv16 = torch.stack([uu16, vv16], dim=0).unsqueeze(0)  # (1, 2, 16, 16)
        self.register_buffer("uv_grid_16", uv16)
        self.register_buffer("uv_flat_16", uv16.flatten(2).transpose(1, 2).contiguous())  # (1, 256, 2)

        self.pos_proj = nn.Linear(2, OBJECT_SLOT_DIM)
        self.slot_queries = nn.Parameter(torch.randn(1, NUM_OBJECT_SLOTS, OBJECT_SLOT_DIM) * 0.25)
        self.slot_gru = nn.GRUCell(OBJECT_SLOT_DIM, OBJECT_SLOT_DIM)
        self.slot_CoordHead = nn.Linear(OBJECT_SLOT_DIM, 3)     # [u, v, depth_norm]
        self.slot_ValenceHead = nn.Linear(OBJECT_SLOT_DIM, 3)   # [hostile, objective, turf_or_cover]

        self.out_z_proj = nn.Linear(OBJECT_SLOT_DIM * 3, LATENT_Z_DIM)
        nn.init.orthogonal_(self.out_z_proj.weight, gain=0.25)
        nn.init.zeros_(self.out_z_proj.bias)
        self.z_res_gate = nn.Parameter(torch.zeros(1))

        # Bidirectional GLW Top-Down Generative Slot Prior (`(prev_gwt, prev_rssm_h) -> slot_queries` prior)
        # Zero-gated (`td_slot_gate = 0` at init) so legacy checkpoints load with 0.000000 perturbation!
        self.td_slot_prior = nn.Sequential(
            nn.Linear(GWT_DIM + RSSM_DETER_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, NUM_OBJECT_SLOTS * OBJECT_SLOT_DIM),
        )
        nn.init.orthogonal_(self.td_slot_prior[-1].weight, gain=0.20)
        nn.init.zeros_(self.td_slot_prior[-1].bias)
        self.td_slot_gate = nn.Parameter(torch.zeros(1))

        self.next_slot_pred = nn.Sequential(
            nn.Linear(NUM_OBJECT_SLOTS * OBJECT_SLOT_DIM + 32, 96),
            nn.GELU(),
            nn.Linear(96, NUM_OBJECT_SLOTS * OBJECT_SLOT_DIM),
        )
        self.rssm_slot_ctx = nn.Linear(RSSM_DETER_DIM, NUM_OBJECT_SLOTS * OBJECT_SLOT_DIM)
        nn.init.orthogonal_(self.rssm_slot_ctx.weight, gain=0.15)
        nn.init.zeros_(self.rssm_slot_ctx.bias)
        self.rssm_slot_gate = nn.Parameter(torch.zeros(1))

        self.fg_dest_head = nn.Linear(OBJECT_SLOT_DIM, FOVEA_H * FOVEA_W)
        nn.init.constant_(self.fg_dest_head.bias, -2.0)
        self.slot_app_delta = nn.Sequential(
            nn.Linear(OBJECT_SLOT_DIM + VIS8_C, OBJECT_SLOT_DIM),
            nn.GELU(),
            nn.Linear(OBJECT_SLOT_DIM, VIS8_C),
        )
        nn.init.normal_(self.slot_app_delta[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.slot_app_delta[-1].bias)
        self.bg_spatial_refine = nn.Sequential(
            nn.Conv2d(OBJECT_SLOT_DIM + 2, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, VIS8_C, kernel_size=1),
        )
        nn.init.normal_(self.bg_spatial_refine[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bg_spatial_refine[-1].bias)

    def forward(
        self,
        vis8_bhwc,
        pred_err_16=None,
        topdown_gain_16=None,
        td_context=None,
        return_attn: bool = False,
        return_aux: bool = False,
    ):
        bsz = vis8_bhwc.shape[0]
        dev, dt = vis8_bhwc.device, vis8_bhwc.dtype
        x = vis8_bhwc.permute(0, 3, 1, 2).contiguous()  # (B, 8, 16, 16)
        if pred_err_16 is None:
            pred_err_16 = torch.zeros((bsz, 1, FOVEA_H, FOVEA_W), device=dev, dtype=dt)
        if topdown_gain_16 is None:
            topdown_gain_16 = torch.ones((bsz, 1, FOVEA_H, FOVEA_W), device=dev, dtype=dt)
        gain_c = torch.clamp(topdown_gain_16, 0.50, 1.60)

        err_spot_in = torch.cat([torch.clamp(pred_err_16, 0.0, 1.5), gain_c - 1.0], dim=1)
        feat16 = self.backbone[1](self.backbone[0](x * gain_c) + self.err_spot_conv(err_spot_in))  # (B, 32, 16, 16)
        feat8 = self.backbone[3](self.backbone[2](feat16))  # (B, 32, 8, 8)
        feat = feat8.flatten(2).permute(0, 2, 1) + self.pos_proj(self.grid_xy)  # (B, 64, 32)

        # Bottom-up + prediction-error + spotlight salience prior separating Slot 0 (BG) vs Slots 1..3 (FG)
        d_ch = x[:, 0:1]
        d_bg = F.avg_pool2d(d_ch, kernel_size=5, stride=1, padding=2)
        protrusion = F.relu(d_ch - d_bg) * 2.0
        chroma_pop = 0.60 * (torch.abs(x[:, 4:5]) + torch.abs(x[:, 5:6]))
        motion_pop = 0.55 * torch.clamp(x[:, 7:8], 0.0, 1.0)
        err_pop = 0.75 * torch.clamp(pred_err_16, 0.0, 1.0)
        sal_16 = torch.clamp((protrusion + chroma_pop + motion_pop + err_pop) * gain_c, 0.0, 2.5)
        sal_8 = F.avg_pool2d(sal_16, kernel_size=2, stride=2).flatten(1)  # (B, 64)
        bg_bias = (1.10 - 2.60 * sal_8).unsqueeze(1)  # (B, 1, 64)
        fg_bias = (1.50 * sal_8 - 0.60).unsqueeze(1).expand(-1, NUM_OBJECT_SLOTS - 1, -1)  # (B, 3, 64)
        prior_bias = torch.cat([bg_bias, fg_bias], dim=1)  # (B, 4, 64)

        slots = self.slot_queries.expand(bsz, -1, -1)
        td_prior_raw = None
        if td_context is not None:
            td_prior_raw = self.td_slot_prior(td_context).view(bsz, NUM_OBJECT_SLOTS, OBJECT_SLOT_DIM)
            slots = slots + 0.45 * torch.tanh(self.td_slot_gate) * torch.tanh(td_prior_raw)
        last_attn = None
        for _ in range(2):
            attn_logits = torch.einsum("bsd,bnd->bsn", slots, feat) / math.sqrt(float(OBJECT_SLOT_DIM)) + prior_bias
            attn = F.softmax(attn_logits, dim=1)  # Competitive across slots!
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
            last_attn = attn
            updates = torch.einsum("bsn,bnd->bsd", attn, feat)
            slots = self.slot_gru(
                updates.reshape(-1, OBJECT_SLOT_DIM),
                slots.reshape(-1, OBJECT_SLOT_DIM),
            ).reshape(bsz, NUM_OBJECT_SLOTS, OBJECT_SLOT_DIM)
        coords = torch.tanh(self.slot_CoordHead(slots))
        valence = torch.sigmoid(self.slot_ValenceHead(slots))

        vis8_8x8 = F.avg_pool2d(x, kernel_size=2, stride=2).flatten(2).permute(0, 2, 1)  # (B, 64, 8)
        pixel_stats = torch.einsum("bsn,bnd->bsd", last_attn, vis8_8x8)  # (B, 4, 8)
        fg_slots = slots[:, 1:]
        fg_pool = torch.cat([fg_slots.mean(dim=1), fg_slots.amax(dim=1), slots[:, 0]], dim=-1)
        delta_z = torch.tanh(self.z_res_gate) * self.out_z_proj(fg_pool)

        if return_aux:
            return slots, coords, valence, {
                "attn": last_attn,
                "feat16": feat16,
                "sal_16": sal_16,
                "pixel_stats": pixel_stats,
                "delta_z": delta_z,
                "td_prior_raw": td_prior_raw,
            }
        if return_attn:
            return slots, coords, valence, last_attn
        return slots, coords, valence

    def predict_next_slots(self, slots, cand_feat32, rssm_h=None):
        bsz = slots.shape[0]
        flat = slots.reshape(bsz, -1)
        pred_flat = self.next_slot_pred(torch.cat([flat, cand_feat32], dim=-1))
        if rssm_h is not None:
            pred_flat = pred_flat + torch.tanh(self.rssm_slot_gate) * self.rssm_slot_ctx(rssm_h)
        return (slots + 0.50 * pred_flat.reshape(bsz, NUM_OBJECT_SLOTS, OBJECT_SLOT_DIM))

    def predict_next_visual_field(self, slots, pixel_stats, coords, cand_feat32, rssm_h=None):
        bsz = slots.shape[0]
        dev, dt = slots.device, slots.dtype
        next_slots = self.predict_next_slots(slots, cand_feat32, rssm_h=rssm_h)
        d_app = 0.65 * torch.tanh(self.slot_app_delta(torch.cat([next_slots, pixel_stats], dim=-1)))
        slot_vis = pixel_stats + d_app  # (B, 4, 8)

        bg_tile = next_slots[:, 0].view(bsz, OBJECT_SLOT_DIM, 1, 1).expand(-1, -1, FOVEA_H, FOVEA_W)
        bg_ref = 0.12 * torch.tanh(
            self.bg_spatial_refine(torch.cat([bg_tile, self.uv_grid_16.expand(bsz, -1, -1, -1)], dim=1))
        )
        bg_canvas = slot_vis[:, 0].view(bsz, VIS8_C, 1, 1) + bg_ref  # (B, 8, 16, 16)

        fg_logits = self.fg_dest_head(next_slots[:, 1:]).view(bsz, NUM_OBJECT_SLOTS - 1, FOVEA_H, FOVEA_W)
        fg_logits = F.avg_pool2d(
            fg_logits.reshape(-1, 1, FOVEA_H, FOVEA_W), kernel_size=3, stride=1, padding=1
        ).view(bsz, NUM_OBJECT_SLOTS - 1, FOVEA_H, FOVEA_W)
        if coords is not None:
            uv_f = self.uv_flat_16.expand(bsz, -1, -1).unsqueeze(1)  # (B, 1, 256, 2)
            c_uv = coords[:, 1:, :2].unsqueeze(2)  # (B, 3, 1, 2)
            dist_sq = ((uv_f - c_uv) ** 2).sum(dim=-1).view(bsz, NUM_OBJECT_SLOTS - 1, FOVEA_H, FOVEA_W)
            fg_logits = fg_logits - 3.0 * F.relu(dist_sq - 0.45)

        bg_logit = torch.zeros((bsz, 1, FOVEA_H, FOVEA_W), device=dev, dtype=dt)
        alpha = F.softmax(torch.cat([bg_logit, fg_logits], dim=1), dim=1).unsqueeze(2)  # (B, 4, 1, 16, 16)
        fg_canvases = slot_vis[:, 1:].view(bsz, NUM_OBJECT_SLOTS - 1, VIS8_C, 1, 1).expand(
            -1, -1, -1, FOVEA_H, FOVEA_W
        )
        comp = (alpha * torch.cat([bg_canvas.unsqueeze(1), fg_canvases], dim=1)).sum(dim=1)  # (B, 8, 16, 16)

        pos_raw = comp[:, [0, 3, 6, 7]]
        pos_c = pos_raw.clamp(0.0, 1.0)
        pos_ch = pos_c + 0.10 * ((pos_raw - pos_c) - (pos_raw - pos_c).detach())
        signed_raw = comp[:, [1, 2, 4, 5]]
        signed_c = signed_raw.clamp(-1.0, 1.0)
        signed_ch = signed_c + 0.10 * ((signed_raw - signed_c) - (signed_raw - signed_c).detach())
        pred_bchw = torch.stack(
            [
                pos_ch[:, 0], signed_ch[:, 0], signed_ch[:, 1], pos_ch[:, 1],
                signed_ch[:, 2], signed_ch[:, 3], pos_ch[:, 2], pos_ch[:, 3],
            ],
            dim=1,
        )
        return pred_bchw.permute(0, 2, 3, 1).contiguous(), next_slots

class SpatialTopologicalGWTAdapter3D(nn.Module):
    """
    Fuses 4 Object Slots with `MEC`/`HD`/`Compass` into `64D` topological context AND computes
    Zero-Gated GWT Spatial Cross-Attention + `16x16` Top-Down Precision Spotlight (`topdown_gain_16`).
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(NUM_OBJECT_SLOTS * OBJECT_SLOT_DIM + 64, MULTIMODAL_EMBED_DIM),
            nn.LayerNorm(MULTIMODAL_EMBED_DIM),
            nn.GELU(),
        )
        self.sp_comp = nn.Sequential(
            nn.Linear(MEC_GRID_DIM + HD_RING_DIM + MEM_COMPASS_DIM, 64),
            nn.GELU(),
        )
        self.grid_proj = nn.Linear(OBJECT_SLOT_DIM + 2, 64)
        self.slot_proj = nn.Linear(OBJECT_SLOT_DIM + 3, 64)
        self.kv_norm = nn.LayerNorm(64)
        self.gwt_norm = nn.LayerNorm(GWT_DIM)
        self.q_proj = nn.Linear(GWT_DIM, 64, bias=False)
        self.v_proj = nn.Linear(64, 64, bias=False)
        self.out_gwt_proj = nn.Linear(64, GWT_DIM)
        nn.init.orthogonal_(self.out_gwt_proj.weight, gain=0.25)
        nn.init.zeros_(self.out_gwt_proj.bias)
        self.gwt_res_gate = nn.Parameter(torch.zeros(1))

        self.spot_q_proj = nn.Linear(GWT_DIM, OBJECT_SLOT_DIM)
        self.spot_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.spot_conv.weight, gain=0.25)
        nn.init.constant_(self.spot_conv.bias, 0.10)
        self.spot_gate = nn.Parameter(torch.zeros(1))

        u4 = torch.linspace(-1.0, 1.0, 4)
        v4 = torch.linspace(-1.0, 1.0, 4)
        vv4, uu4 = torch.meshgrid(v4, u4, indexing="ij")
        self.register_buffer("uv4_flat", torch.stack([uu4, vv4], dim=-1).reshape(1, 16, 2))

    def forward(self, slots, mec_code, hd_code, mem_compass):
        bsz = slots.shape[0]
        sp_e = self.sp_comp(torch.cat([mec_code, hd_code, mem_compass], dim=-1))
        return self.proj(torch.cat([slots.reshape(bsz, -1), sp_e], dim=-1))

    def compute_topdown_spotlight(self, gwt_vec, feat16_bchw, slots, coords):
        bsz = gwt_vec.shape[0]
        g4 = F.adaptive_avg_pool2d(feat16_bchw, (4, 4)).flatten(2).transpose(1, 2).contiguous()  # (B, 16, 32)
        grid_tok = self.grid_proj(torch.cat([g4, self.uv4_flat.expand(bsz, -1, -1)], dim=-1))
        slot_tok = self.slot_proj(torch.cat([slots, coords], dim=-1))
        kv_tok = self.kv_norm(torch.cat([grid_tok, slot_tok], dim=1))  # (B, 20, 64)
        q = self.q_proj(self.gwt_norm(gwt_vec)).unsqueeze(1)  # (B, 1, 64)
        attn_w = F.softmax((q @ kv_tok.transpose(1, 2)) * 0.125, dim=-1).squeeze(1)  # (B, 20)
        ctx_64 = (attn_w.unsqueeze(-1) * self.v_proj(kv_tok)).sum(dim=1)
        delta_gwt = torch.tanh(self.gwt_res_gate) * self.out_gwt_proj(ctx_64)

        spot_q = self.spot_q_proj(gwt_vec + delta_gwt).view(bsz, OBJECT_SLOT_DIM, 1, 1)
        feat_dot = (spot_q * feat16_bchw).sum(dim=1, keepdim=True) / math.sqrt(float(OBJECT_SLOT_DIM))
        attn_16 = F.interpolate(attn_w[:, :16].view(bsz, 1, 4, 4), size=(FOVEA_H, FOVEA_W), mode="bilinear", align_corners=False)
        raw_spot = self.spot_conv(torch.cat([feat_dot, attn_16], dim=1))
        topdown_gain_16 = 1.0 + 0.50 * torch.tanh(self.spot_gate) * torch.tanh(raw_spot)
        return delta_gwt, topdown_gain_16

class SlotConceptLanguageGrounder3D(nn.Module):
    """
    Contrastive InfoNCE + Foreground-Slot MIL-InfoNCE Multimodal Concept Grounder (`v2.1`):
      - Aligns global `slot_ctx64` AND translation-invariant foreground visual slots (`Slots 1..3`)
        with Wernicke language & RVQ Proto-Symbols while repelling background `Slot 0` (hard negative).
    """

    def __init__(self):
        super().__init__()
        self.vis_proj = nn.Linear(MULTIMODAL_EMBED_DIM, INNER_SPEECH_EMBED_DIM)
        self.lang_proj = nn.Linear(INNER_SPEECH_EMBED_DIM, INNER_SPEECH_EMBED_DIM)
        what_in_dim = VIS8_C * 2 + 2  # 18D translation-invariant ventral slot stats (excludes u, v & MEC!)
        self.register_buffer(
            "fg_ind",
            torch.tensor([0.0, 1.0, 1.0, 1.0], dtype=torch.float32).view(1, NUM_OBJECT_SLOTS, 1),
        )
        self.slot_what_proj = nn.Sequential(
            nn.Linear(what_in_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Linear(48, INNER_SPEECH_EMBED_DIM),
        )
        self.proto_concept_emb = nn.Embedding(NUM_PROTO_SYMBOLS, INNER_SPEECH_EMBED_DIM)

    def encode_visual_slots(self, slot_pstats, slot_coords):
        bsz = slot_pstats.shape[0]
        depth_conf = torch.clamp(0.5 * (slot_coords[:, :, 2:3] + 1.0), 0.0, 1.0)
        fg_m = self.fg_ind.expand(bsz, -1, -1).to(dtype=slot_pstats.dtype) * depth_conf
        what_x = torch.cat(
            [slot_pstats * 2.5, torch.abs(slot_pstats) * 2.0, depth_conf, fg_m],
            dim=-1,
        )
        return F.normalize(self.slot_what_proj(what_x), dim=-1, eps=1e-6)  # (B, 4, 32)

    def slot_mil_loss(self, slot_pstats, slot_coords, l_norm, proto_ids=None, pos_mask=None, temp: float = 0.14):
        bsz = int(slot_pstats.shape[0])
        dev = slot_pstats.device
        v_slots = self.encode_visual_slots(slot_pstats, slot_coords)  # (B, 4, 32)
        v_bg = v_slots[:, 0]      # (B, 32) Slot 0 background
        v_fg = v_slots[:, 1:]     # (B, 3, 32) Slots 1..3 foreground

        fg_depth = torch.clamp(0.5 * (slot_coords[:, 1:, 2] + 1.0), 0.0, 1.0)  # (B, 3)
        fg_sims = torch.einsum("bkd,bd->bk", v_fg, l_norm)  # (B, 3)
        fg_score = fg_sims + 0.25 * (fg_depth - fg_depth.mean(dim=-1, keepdim=True))
        tau = float(max(0.05, temp))
        pos_sim = 0.65 * fg_score.max(dim=-1).values + 0.35 * tau * torch.logsumexp(fg_score / tau, dim=-1)

        # Slot 0 background hard-negative margin
        bg_sim = (v_bg * l_norm).sum(dim=-1)
        loss_bg_margin = F.relu(bg_sim - fg_sims.max(dim=-1).values + 0.35).mean()

        # Cross-batch + Slot 0 MIL-InfoNCE
        loss_mil_nce = l_norm.new_tensor(0.0)
        if bsz >= 2:
            other_neg = ~(pos_mask if pos_mask is not None else torch.eye(bsz, dtype=torch.bool, device=dev))
            bg_valid = torch.ones((bsz, bsz, 1), dtype=torch.bool, device=dev)
            fg_valid = other_neg.unsqueeze(-1).expand(-1, -1, NUM_OBJECT_SLOTS - 1)
            valid_neg = torch.cat([bg_valid, fg_valid], dim=-1).reshape(bsz, bsz * NUM_OBJECT_SLOTS)
            all_neg_logits = (l_norm @ v_slots.reshape(bsz * NUM_OBJECT_SLOTS, INNER_SPEECH_EMBED_DIM).T) / tau
            mil_logits = torch.cat(
                [(pos_sim / tau).unsqueeze(1), all_neg_logits.masked_fill(~valid_neg, -1e4)],
                dim=1,
            )
            zeros_b = torch.zeros(bsz, dtype=torch.long, device=dev)
            loss_mil_nce = F.cross_entropy(mil_logits, zeros_b)
        else:
            loss_mil_nce = (1.0 - pos_sim).mean()

        loss_proto = 0.01 * self.proto_concept_emb.weight.pow(2).mean()
        if proto_ids is not None:
            p_cl = proto_ids.long().clamp(0, NUM_PROTO_SYMBOLS - 1)
            p_emb = F.normalize(self.proto_concept_emb(p_cl), dim=-1, eps=1e-6)
            b_ar = torch.arange(bsz, device=dev)
            best_fg = torch.argmax(fg_score, dim=-1)
            fg_best_v = v_fg[b_ar, best_fg]
            loss_proto = loss_proto + (1.0 - (fg_best_v * p_emb).sum(dim=-1)).mean() + 0.50 * F.relu(
                (v_bg * p_emb).sum(dim=-1) + 0.15
            ).mean()

        return loss_mil_nce + 0.85 * loss_bg_margin + 0.35 * loss_proto

    def alignment_loss(
        self,
        slot_ctx64,
        lang_emb32,
        token_ids=None,
        temp: float = 0.15,
        slot_pstats=None,
        slot_coords=None,
        proto_ids=None,
    ):
        v = F.normalize(self.vis_proj(slot_ctx64), dim=-1, eps=1e-6)
        l = F.normalize(self.lang_proj(lang_emb32), dim=-1, eps=1e-6)
        w_v = F.normalize(self.vis_proj.weight, dim=-1, eps=1e-6)
        w_l = F.normalize(self.lang_proj.weight, dim=-1, eps=1e-6)
        eye_d = torch.eye(INNER_SPEECH_EMBED_DIM, device=v.device, dtype=v.dtype)
        ortho_reg = (w_v @ w_v.transpose(0, 1) - eye_d).pow(2).mean() + (w_l @ w_l.transpose(0, 1) - eye_d).pow(2).mean()

        bsz = int(v.shape[0])
        pos_mask = torch.eye(max(1, bsz), dtype=torch.bool, device=v.device)
        if bsz <= 1:
            base_loss = (1.0 - (v * l).sum(dim=-1).mean()) + 0.05 * ortho_reg
        else:
            logits = torch.matmul(v, l.transpose(0, 1)) / float(max(0.05, temp))
            if token_ids is not None and int(token_ids.shape[0]) == bsz:
                same_tok = (token_ids.unsqueeze(1) == token_ids.unsqueeze(0)).all(dim=-1)
                if not bool(same_tok.all().item()):
                    pos_mask = pos_mask | same_tok

            pos_f = pos_mask.to(logits.dtype)
            log_p_vl = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            loss_vl = -(log_p_vl * pos_f).sum(dim=-1) / pos_f.sum(dim=-1).clamp(min=1.0)

            logits_t = logits.transpose(0, 1)
            pos_ft = pos_f.transpose(0, 1)
            log_p_lv = logits_t - torch.logsumexp(logits_t, dim=-1, keepdim=True)
            loss_lv = -(log_p_lv * pos_ft).sum(dim=-1) / pos_ft.sum(dim=-1).clamp(min=1.0)
            base_loss = 0.5 * (loss_vl.mean() + loss_lv.mean()) + 0.05 * ortho_reg

        if slot_pstats is not None and slot_coords is not None:
            mil_l = self.slot_mil_loss(
                slot_pstats, slot_coords, l, proto_ids=proto_ids, pos_mask=pos_mask if bsz >= 2 else None, temp=temp
            )
            return base_loss + 0.65 * mil_l
        return base_loss

class ActiveGWTWorkspace3D(nn.Module):
    """
    5-Specialist Global Workspace with Nonlinear All-or-None Ignition Phase-Transition Gate (`v2.1`):
      - Sub-threshold diffuse competition preserves reverberant `prev_gwt` working memory.
      - Supra-threshold specialist coalition triggers steep sigmoid global ignition (`ignition`),
        writing strongly to `gwt_w` and scaling top-down cortical modulation `z_td`.
    """

    def __init__(self):
        super().__init__()
        self.s0_vis = nn.Linear(LATENT_Z_DIM + MULTIMODAL_EMBED_DIM, GWT_DIM)
        self.s1_spa = nn.Linear(MEC_GRID_DIM + HD_RING_DIM + MEM_COMPASS_DIM, GWT_DIM)
        self.s2_rssm = nn.Linear(RSSM_DETER_DIM + RSSM_STOCH_FLAT, GWT_DIM)
        self.s3_prop = nn.Linear(PROPRIO_DIM + 1, GWT_DIM)
        self.s4_strat = nn.Linear(64, GWT_DIM)

        self.salience_scorer = nn.Sequential(nn.Linear(GWT_DIM, 32), nn.Tanh(), nn.Linear(32, 1))
        self.broadcast_norm = nn.LayerNorm(GWT_DIM)
        self.topdown_z = nn.Sequential(nn.Linear(GWT_DIM, LATENT_Z_DIM), nn.Tanh())
        self.ign_thresh_logit = nn.Parameter(torch.zeros(1))
        self.ign_slope_raw = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        z,
        slot_ctx64,
        mec_code,
        hd_code,
        mem_compass,
        rssm_h,
        rssm_stoch,
        proprio,
        haz_pred,
        strat_emb64,
        prev_gwt,
        meta_salience=None,
        hab_trace=None,
        novelty_5=None,
    ):
        sp0 = F.gelu(self.s0_vis(torch.cat([z, slot_ctx64], dim=-1)))
        sp1 = F.gelu(self.s1_spa(torch.cat([mec_code, hd_code, mem_compass], dim=-1)))
        sp2 = F.gelu(self.s2_rssm(torch.cat([rssm_h, rssm_stoch], dim=-1)))
        sp3 = F.gelu(self.s3_prop(torch.cat([proprio, haz_pred.unsqueeze(-1)], dim=-1)))
        sp4 = F.gelu(self.s4_strat(strat_emb64))

        specialists = torch.stack([sp0, sp1, sp2, sp3, sp4], dim=1)  # (B, 5, 128)
        scores = self.salience_scorer(specialists).squeeze(-1)  # (B, 5)
        eff_scores = scores
        if hab_trace is not None:
            # Tsodyks-Markram short-term synaptic depression on persistently dominant specialist,
            # dishabituated immediately when fresh prediction-error novelty arrives!
            fatigue = F.relu(hab_trace - 0.20)
            if novelty_5 is not None:
                fatigue = fatigue * torch.exp(-1.6 * torch.clamp(novelty_5, 0.0, 2.0))
            eff_scores = eff_scores - 2.45 * fatigue
        if novelty_5 is not None:
            eff_scores = eff_scores + 1.15 * torch.tanh(torch.clamp(novelty_5, 0.0, 2.5))

        alpha = F.softmax(eff_scores * 1.6, dim=-1)
        pooled = torch.einsum("bs,bsd->bd", alpha, specialists)

        max_a = alpha.max(dim=-1).values
        ent = -(alpha * torch.log(alpha + 1e-7)).sum(dim=-1) / math.log(5.0)
        ign_coherence = torch.clamp(0.65 * max_a + 0.35 * (1.0 - ent), 0.0, 1.0)

        kappa = 8.0 + 8.0 * torch.sigmoid(self.ign_slope_raw)  # ~12.0 at init
        if hab_trace is not None or novelty_5 is not None:
            # Metastable Phasic Ignition: quiet reverberant maintenance (~0.20-0.42) during steady cruising,
            # sharp supra-threshold global ignition (~0.75-0.96) upon coalition handoff or surprise!
            pooled_norm = self.broadcast_norm(pooled)
            prev_n = prev_gwt.norm(dim=-1)
            cos_sim = F.cosine_similarity(pooled_norm, prev_gwt, dim=-1, eps=1e-5)
            ws_shift = torch.where(prev_n > 1e-3, torch.clamp(0.5 * (1.0 - cos_sim) * 2.2, 0.0, 1.0), torch.full_like(prev_n, 0.72))
            phasic_nov = (
                torch.tanh(torch.clamp(novelty_5.max(dim=-1).values, 0.0, 2.5))
                if novelty_5 is not None
                else (torch.tanh(torch.clamp(meta_salience, 0.0, 2.0)) if meta_salience is not None else torch.zeros_like(prev_n))
            )
            ign_drive = 0.34 * ign_coherence + 0.38 * ws_shift + 0.28 * phasic_nov
            theta_meta = 0.35 + 0.16 * torch.sigmoid(self.ign_thresh_logit)
            if meta_salience is not None:
                theta_meta = theta_meta - 0.05 * torch.tanh(torch.clamp(meta_salience, 0.0, 2.0))
            ignition = torch.sigmoid((0.65 * kappa) * (ign_drive - theta_meta))
        else:
            theta_base = 0.18 + 0.22 * torch.sigmoid(self.ign_thresh_logit)
            if meta_salience is not None:
                theta_eff = theta_base - 0.08 * torch.tanh(torch.clamp(meta_salience, 0.0, 2.0))
            else:
                theta_eff = theta_base
            ignition = torch.sigmoid(kappa * (ign_coherence - theta_eff))

        write_gate = (0.12 + 0.76 * ignition).unsqueeze(-1)
        gwt_w = self.broadcast_norm((1.0 - write_gate) * prev_gwt + write_gate * pooled)
        z_td = (0.08 + 0.24 * ignition).unsqueeze(-1) * self.topdown_z(gwt_w)
        return gwt_w, alpha, ignition, z_td

class CoarseSelfModel3D(nn.Module):
    """
    HOT / Attention-Schema Metacognitive Self-Model (`CoarseSelfModel3D` v2.1):
      - Encodes current internal self-state from `[gwt_w, proprio]` + Attention Schema `gwt_alpha[5]`
      - Predicts expected self-state from `prev_self` + Efference Copy `[prev_act(32), prev_alpha(5)]`
      - Outputs `(cur, surprise, err_vec, pred)` to close the metacognitive self-prediction error loop!
    """

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(GWT_DIM + PROPRIO_DIM, 48),
            nn.GELU(),
            nn.Linear(48, SELF_MODEL_DIM),
        )
        self.attn_schema_proj = nn.Linear(5, SELF_MODEL_DIM, bias=False)
        nn.init.orthogonal_(self.attn_schema_proj.weight, gain=0.25)
        self.schema_gate = nn.Parameter(torch.zeros(1))

        self.pred_next = nn.Linear(SELF_MODEL_DIM, SELF_MODEL_DIM)
        self.efference_proj = nn.Sequential(
            nn.Linear(32 + 5, 24),
            nn.GELU(),
            nn.Linear(24, SELF_MODEL_DIM),
        )
        nn.init.normal_(self.efference_proj[-1].weight, mean=0.0, std=0.03)
        nn.init.zeros_(self.efference_proj[-1].bias)

    def forward(self, gwt_w, proprio, prev_self, gwt_alpha=None, prev_act=None, prev_alpha=None):
        bsz = gwt_w.shape[0]
        dev, dt = gwt_w.device, gwt_w.dtype
        if gwt_alpha is None:
            gwt_alpha = torch.full((bsz, 5), 0.2, device=dev, dtype=dt)
        if prev_act is None:
            prev_act = torch.zeros((bsz, 32), device=dev, dtype=dt)
        if prev_alpha is None:
            prev_alpha = torch.full((bsz, 5), 0.2, device=dev, dtype=dt)

        raw_s = self.enc(torch.cat([gwt_w, proprio], dim=-1))
        cur = torch.tanh(raw_s + torch.tanh(self.schema_gate) * self.attn_schema_proj(gwt_alpha))
        eff_in = torch.cat([prev_act, prev_alpha], dim=-1)
        pred = torch.tanh(self.pred_next(prev_self) + self.efference_proj(eff_in))
        err_vec = cur.detach() - pred
        surprise = torch.mean(err_vec ** 2, dim=-1)
        return cur, surprise, err_vec, pred

class StrategicTransformer3D(nn.Module):
    """
    Prefrontal Episodic Context Transformer (`StrategicTransformer3D` v2.1):
    Attends over recent `(gwt_workspace, self_state, mem_compass)` snapshots + causal metacognitive
    `self_surprise` error residual (`meta_err9`) and outputs:
      - `strat_embed (B, 64)` for GWT Specialist #4
      - `goal_mode_logits (B, 5)` selecting the active macro tactical mode!
      - `focus_offset_xz (B, 2)` steering top-down spatial gaze lookahead!
    """

    def __init__(self):
        super().__init__()
        in_d = GWT_DIM + SELF_MODEL_DIM + MEM_COMPASS_DIM  # 160
        self.in_proj = nn.Linear(in_d, 64)
        self.pos_emb = nn.Parameter(torch.zeros(1, 8, 64))
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((8, 8), float("-inf")), diagonal=1),
            persistent=False,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.meta_err_proj = nn.Linear(SELF_MODEL_DIM + 1, 64)
        nn.init.orthogonal_(self.meta_err_proj.weight, gain=0.25)
        nn.init.zeros_(self.meta_err_proj.bias)
        self.meta_err_gate = nn.Parameter(torch.zeros(1))
        self.goal_mode_head = nn.Linear(64, 5)
        self.focus_offset_head = nn.Sequential(nn.Linear(64, 2), nn.Tanh())

    def forward(self, hist_seq, meta_err9=None):
        # hist_seq: (B, K, 160) with zero-initialized learnable positional encoding + causal attention mask
        seq_len = int(hist_seq.shape[1])
        if seq_len <= int(self.pos_emb.shape[1]):
            pos_slice = self.pos_emb[:, :seq_len, :]
        else:
            pos_slice = F.interpolate(
                self.pos_emb.permute(0, 2, 1), size=seq_len, mode="linear", align_corners=False
            ).permute(0, 2, 1)
        h = self.in_proj(hist_seq) + pos_slice
        if seq_len <= int(self.causal_mask.shape[0]):
            c_mask = self.causal_mask[:seq_len, :seq_len].to(dtype=h.dtype)
        else:
            c_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=h.device, dtype=h.dtype), diagonal=1
            )
        out = self.transformer(h, mask=c_mask)[:, -1, :]  # (B, 64)
        if meta_err9 is not None:
            out = out + torch.tanh(self.meta_err_gate) * F.gelu(self.meta_err_proj(meta_err9))
        mode_logits = self.goal_mode_head(out)
        focus_off = self.focus_offset_head(out) * 5.0  # [-5m..+5m] lookahead offset
        return out, mode_logits, focus_off

class EndogenousBrocaWernickeCortex(nn.Module):
    """
    Vygotskian Wernicke Comprehension + Broca Grounded Telemetry Generator.
    Under Plan A (`cfg.allow_broca_motor_bias == False`), `cand_bias` is strictly zeroed during
    action selection so inner speech can NEVER induce behavioral freeze or corrupt Q-values.
    """

    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(ENDOGENOUS_VOCAB_SIZE, INNER_SPEECH_EMBED_DIM, padding_idx=0)
        self.wernicke_gru = nn.GRU(INNER_SPEECH_EMBED_DIM, INNER_SPEECH_EMBED_DIM, batch_first=True)
        self.ctx_init = nn.Sequential(
            nn.Linear(GWT_DIM + PROTO_EMBED_DIM + MULTIMODAL_EMBED_DIM, 64),
            nn.GELU(),
            nn.Linear(64, INNER_SPEECH_EMBED_DIM),
            nn.Tanh(),
        )
        self.broca_gru = nn.GRUCell(INNER_SPEECH_EMBED_DIM, INNER_SPEECH_EMBED_DIM)
        self.vocab_head = nn.Linear(INNER_SPEECH_EMBED_DIM, ENDOGENOUS_VOCAB_SIZE)
        self.cand_bias_head = nn.Sequential(
            nn.Linear(INNER_SPEECH_EMBED_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, NUM_CANDIDATES),
            nn.Tanh(),
        )

    def encode_tokens(self, tok_ids):
        emb = self.tok_emb(tok_ids.clamp(0, ENDOGENOUS_VOCAB_SIZE - 1))
        _, h_n = self.wernicke_gru(emb)
        return h_n[-1]

    def generate_inner_speech(self, gwt_w, proto_emb, slot_ctx64, proto_sym_id: int = 0, alt_first_word: bool = False):
        h = self.ctx_init(torch.cat([gwt_w, proto_emb, slot_ctx64], dim=-1))
        bsz = gwt_w.shape[0]
        cur_tok = torch.full((bsz,), GroundedVocabulary512.BOS_ID, dtype=torch.long, device=gwt_w.device)
        out_tokens = [cur_tok]
        proto_tok = torch.full(
            (bsz,), GroundedVocabulary512.PROTO_OFFSET + (int(proto_sym_id) % 32), dtype=torch.long, device=gwt_w.device
        )
        h = self.broca_gru(self.tok_emb(cur_tok), h)
        out_tokens.append(proto_tok)
        cur_tok = proto_tok
        logp_acc = gwt_w.new_zeros((bsz,))
        for step_k in range(INNER_SPEECH_MAX_LEN - 2):
            h = self.broca_gru(self.tok_emb(cur_tok), h)
            logits = self.vocab_head(h)
            eos_val = logits[:, GroundedVocabulary512.EOS_ID].clone()
            logits[:, :36] = -1e4   # Restrict Broca output to grounded tactical CORE_WORDS (36..83) + EOS
            logits[:, 84:] = -1e4
            logits[:, GroundedVocabulary512.EOS_ID] = eos_val - 0.50
            lprobs = F.log_softmax(logits, dim=-1)
            if alt_first_word and step_k == 0:
                top2 = torch.topk(logits, k=2, dim=-1).indices
                cur_tok = top2[:, 1]
            else:
                cur_tok = torch.argmax(logits, dim=-1)
            logp_acc = logp_acc + lprobs.gather(-1, cur_tok.unsqueeze(-1)).squeeze(-1)
            out_tokens.append(cur_tok)
        seq = torch.stack(out_tokens, dim=1)  # (B, 12)
        raw_bias = 0.25 * self.cand_bias_head(h)
        return seq, h, raw_bias, logp_acc / float(max(1, INNER_SPEECH_MAX_LEN - 2))

    def teacher_forcing_loss(self, gwt_w, proto_emb, slot_ctx64, target_seq):
        h = self.ctx_init(torch.cat([gwt_w, proto_emb, slot_ctx64], dim=-1))
        loss = gwt_w.new_tensor(0.0)
        steps = 0
        for t in range(target_seq.shape[1] - 1):
            inp = target_seq[:, t].clamp(0, ENDOGENOUS_VOCAB_SIZE - 1)
            tgt = target_seq[:, t + 1].clamp(0, ENDOGENOUS_VOCAB_SIZE - 1)
            h = self.broca_gru(self.tok_emb(inp), h)
            mask = tgt != GroundedVocabulary512.PAD_ID
            if mask.any():
                logits = self.vocab_head(h)
                loss = loss + F.cross_entropy(logits[mask], tgt[mask])
                steps += 1
        return loss / float(max(1, steps)), h

class BinocularRetinotopicDepthNet3D(nn.Module):
    """
    Binocular Stereo Disparity + Multi-Scale Retinotopic Cortex (`BinocularRetinotopicDepthNet3D`):
    Predicts `16x16x4` dorsal geometry `[norm_depth, nx, ny, surface_or_salience]` + `16x16`
    heteroscedastic epistemic uncertainty directly from `2 x 64 x 64 x 3` stereo RGB24 views.
    """

    def __init__(self, max_disp: int = 8):
        super().__init__()
        self.max_disp = int(max_disp)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(20, 24, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )  # -> (B, 24, 16, 16)
        in_ch = 24 * 2 + self.max_disp
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 48, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.geom_head = nn.Conv2d(32, FOVEA_C, kernel_size=1)
        self.logvar_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, stereo_u8):
        x = stereo_u8.to(torch.float32) / 255.0  # (B, 2, 64, 64, 3)
        xl = self.stem(x[:, 0].permute(0, 3, 1, 2).contiguous())
        xr = self.stem(x[:, 1].permute(0, 3, 1, 2).contiguous())
        corr_list = []
        for d in range(self.max_disp):
            if d == 0:
                c = (xl * xr).mean(dim=1, keepdim=True)
            else:
                shifted = F.pad(xr[:, :, :, :-d], (d, 0, 0, 0))
                c = (xl * shifted).mean(dim=1, keepdim=True)
            corr_list.append(c)
        cost_vol = torch.cat(corr_list, dim=1)
        h = self.decoder(torch.cat([xl, xr, cost_vol], dim=1))
        raw_g = self.geom_head(h).permute(0, 2, 3, 1).contiguous()
        d_norm = torch.sigmoid(raw_g[:, :, :, 0:1])
        n_xy = torch.tanh(raw_g[:, :, :, 1:3])
        sal = torch.sigmoid(raw_g[:, :, :, 3:4])
        pred_patch = torch.cat([d_norm, n_xy, sal], dim=-1)
        logvar = torch.clamp(self.logvar_head(h).squeeze(1), -5.0, 2.0)
        uncert = torch.clamp(torch.exp(0.5 * logvar) * 0.35, 0.0, 1.0)
        return pred_patch, uncert, logvar


def build_pytorch_brain_models(torch_mod: Any = None, cfg: Optional[Brain3DConfig] = None) -> Dict[str, nn.Module]:
    """Instantiate the complete 13-model Whole-Brain PyTorch suite in deterministic seed order."""
    _ = (torch_mod, cfg)
    return {
        "foveal_enc": TensorTrainFovealEncoder3D(),
        "hazard_net": HazardPredictor3D(),
        "d3qn": TensorTrainD3QNCandidateNet3D(),
        "d3qn_target": TensorTrainD3QNCandidateNet3D(),
        "rssm": CategoricalRSSM3D(),
        "slot_cortex": ObjectSlotPredictiveCortex3D(),
        "topo_adapter": SpatialTopologicalGWTAdapter3D(),
        "concept_grounder": SlotConceptLanguageGrounder3D(),
        "gwt": ActiveGWTWorkspace3D(),
        "self_model": CoarseSelfModel3D(),
        "strat_tf": StrategicTransformer3D(),
        "broca": EndogenousBrocaWernickeCortex(),
        "depth_net": BinocularRetinotopicDepthNet3D(),
    }
