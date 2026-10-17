import numpy as np
import torch
from einops import rearrange

class PixelDiffusion:

    def __init__(self, config):
        self.args = config
        self.cfg_ratio = config.cfg_ratio
        self.cfg_scale = config.cfg_scale
        self.interp_type = 'lin'
        self.t_start = 0
        self.t_end = 1

    def interpolate(self, cond, uncond, alpha, interp_type='linear'):
        if interp_type == 'sph':
            return alpha * cond + (1 - alpha ** 2) ** 0.5 * uncond
        elif interp_type == 'lin':
            return alpha * cond + (1 - alpha) * uncond

    @torch.no_grad()
    def sample_multistep_fm(self, model, lq, venc_fea=None, n_steps: int=25, schedule: str='linear', guide=None, return_aux: bool=False):
        B, device = (lq.size(0), lq.device)
        observed_lq = lq
        layer_features = venc_fea
        lq_weak = None
        venc_fea_weak = None
        use_cfg = self.cfg_ratio > 0 and int(n_steps) > 1
        if self.args.aug_noise:
            H, W = lq.shape[-2:]
            C = lq.shape[1]
            base_dimension = getattr(self.args, 'time_shift_base_dimension', 4096)
            alpha = np.sqrt(H * W * C / base_dimension)
            eps = alpha * torch.randn_like(lq)
        else:
            eps = torch.randn_like(lq)
        if use_cfg:
            if self.args.use_aelq:
                weak_cond_strength_aelq = (self.args.weak_cond_strength_aelq_list[0] + self.args.weak_cond_strength_aelq_list[1]) / 2.0
                lq_weak = self.interpolate(lq, torch.zeros_like(lq), weak_cond_strength_aelq, self.interp_type)
                lq = self.interpolate(lq, torch.randn_like(lq), self.args.cond_strength_aelq_test, interp_type='sph') if hasattr(self.args, 'cond_strength_aelq_test') else lq
            else:
                lq_weak = torch.zeros_like(lq)
                lq = torch.randn_like(lq)
            if self.args.use_venc:
                if isinstance(venc_fea, (list, tuple)):
                    venc_fea_weak = [self.interpolate(fea, torch.zeros_like(fea), self.args.weak_cond_strength_venc, self.interp_type) for fea in venc_fea]
                    venc_fea = [self.interpolate(fea, torch.randn_like(fea), self.args.cond_strength_venc, interp_type='sph') for fea in venc_fea]
                else:
                    venc_fea_weak = self.interpolate(venc_fea, torch.zeros_like(venc_fea), self.args.weak_cond_strength_venc, self.interp_type)
                    venc_fea = self.interpolate(venc_fea, torch.randn_like(venc_fea), self.args.cond_strength_venc, interp_type='sph')
            else:
                venc_fea_weak = None
        else:
            if self.args.use_aelq:
                lq = self.interpolate(lq, torch.randn_like(lq), self.args.cond_strength_aelq_test, interp_type='sph')
            else:
                lq = torch.randn_like(lq)
            if self.args.use_venc:
                if isinstance(venc_fea, (list, tuple)):
                    venc_fea = [self.interpolate(fea, torch.randn_like(fea), self.args.cond_strength_venc, interp_type='sph') for fea in venc_fea]
                else:
                    venc_fea = self.interpolate(venc_fea, torch.randn_like(venc_fea), self.args.cond_strength_venc, interp_type='sph')
        z = eps
        if schedule == 'linear':
            t_seq = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        elif schedule == 'cosine':
            s = torch.linspace(0, np.pi / 2, n_steps + 1, device=device)
            t_seq = torch.cos(s)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        layer_prediction = None
        for i in range(n_steps):
            t_cur, t_next = (t_seq[i], t_seq[i + 1])
            dt = t_cur - t_next
            model_inp = torch.cat([lq, z], dim=1)
            model_t = t_cur.repeat(B)
            model_venc_fea = venc_fea
            if use_cfg and t_cur <= self.t_end and (t_cur >= self.t_start):
                inp_weak = torch.cat([lq_weak, z], dim=1)
                model_inp = torch.cat([model_inp, inp_weak], dim=0)
                model_t = model_t.repeat(2)
                if venc_fea is not None:
                    if isinstance(venc_fea, (list, tuple)):
                        model_venc_fea = [torch.cat([v, w], dim=0) for v, w in zip(venc_fea, venc_fea_weak)]
                    else:
                        model_venc_fea = torch.cat([venc_fea, venc_fea_weak], dim=0)
            model_kwargs = {'z': model_venc_fea, 'layer_condition': observed_lq, 'layer_features': layer_features}
            if guide is not None:
                model_kwargs['guide'] = torch.cat((guide, guide), dim=0) if model_inp.shape[0] == 2 * B else guide
            x_pred, layer_raw = model(model_inp, model_t, **model_kwargs)
            if i == 0 and torch.is_tensor(layer_raw):
                layer_prediction = layer_raw[:B] if layer_raw.shape[0] == 2 * B else layer_raw
            t_cur_ = rearrange(t_cur, ' -> 1 1 1 1')
            if use_cfg and t_cur <= self.t_end and (t_cur >= self.t_start):
                x_pred_cond, x_pred_weak = x_pred.chunk(2)
                v_pred_cond = (z - x_pred_cond) / t_cur_.clamp_min(self.args.t_eps)
                v_pred_weak = (z - x_pred_weak) / t_cur_.clamp_min(self.args.t_eps)
                v_pred = v_pred_weak + self.cfg_scale * (v_pred_cond - v_pred_weak)
            else:
                v_pred = (z - x_pred) / t_cur_.clamp_min(self.args.t_eps)
            z = z - dt * v_pred
        hq_hat = z
        return (hq_hat, layer_prediction) if return_aux else hq_hat
