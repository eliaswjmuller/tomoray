"""Compare VQGAN reconstruction on the internal head-CT test set:
    old from-scratch (epoch=284, ~0.07)  vs  RSNA clean+tilt -> fine-tuned.
Optionally also the RSNA pretrained models (to show the domain gap they start from).

Test set = the fine-tune's held-out patient-split test (splits_all_bypatient.json['test']),
which the fine-tuned model never saw. NOTE: the old model's original split is unknown,
so it may have trained on some of these -> this is an INDICATIVE head-to-head, not a
perfectly controlled one (that would need a from-scratch run on this same split).

Run: CUDA_VISIBLE_DEVICES=0 python evaluation/eval_internal_compare.py
"""
import os
import sys
import json

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vq_gan_3d.model.vqgan import VQGAN
from data.verse_nifti import VerseDataset
from monai.metrics import SSIMMetric

NIFTI = "/home/user/Desktop/tomoray/datasets/brain_dataset/CTs"
RUNS = "/home/user/Desktop/tomoray/vqgan_runs"
SPLIT = os.path.join(NIFTI, "splits_all_bypatient.json")

CKPTS = {
    "old_from_scratch(ep284)": f"{RUNS}/vqgan_oldbrain_scratch_ep284.ckpt",
    "rsna_clean+tilt->ft":     f"{RUNS}/vqgan_finetune_internal_best.ckpt",
    # context: the pretrained models before any internal fine-tuning
    "rsna_clean+tilt(pre-ft)": f"{RUNS}/vqgan_clean_tilt_best.ckpt",
}


@torch.no_grad()
def eval_model(ckpt, files):
    model = VQGAN.load_from_checkpoint(ckpt, map_location="cuda", weights_only=False).eval().cuda()
    ss = tuple(int(s) for s in model.cfg.dataset.spatial_size)
    ds = VerseDataset(root_dir=NIFTI, split="test", spatial_size=ss, data_list=files)
    ssim = SSIMMetric(spatial_dims=3, data_range=1.0)
    l1s, psnrs, ssims = [], [], []
    for i in range(len(ds)):
        x = ds[i]["image"].unsqueeze(0).cuda().float()
        xr = model(x, evaluation=True)[0].float().clamp(-1, 1)
        l1s.append((x - xr).abs().mean().item())
        x01, xr01 = (x + 1) / 2, (xr + 1) / 2
        mse = ((x01 - xr01) ** 2).mean().item()
        psnrs.append(10.0 * np.log10(1.0 / max(mse, 1e-10)))
        ssims.append(ssim(xr01, x01).mean().item())
    del model
    torch.cuda.empty_cache()
    return len(ds), np.mean(l1s), np.mean(psnrs), np.mean(ssims)


def main():
    files = json.load(open(SPLIT))["test"]
    print(f"internal held-out test (fine-tune patient-split test): {len(files)} files\n")
    hdr = f"{'model':<26} {'N':>3}  {'L1(x4)':>7}  {'L1':>7}  {'PSNR':>6}  {'SSIM':>6}"
    print(hdr); print("-" * len(hdr))
    for name, ck in CKPTS.items():
        if not os.path.exists(ck):
            print(f"{name:<26}  MISSING: {ck}")
            continue
        n, l1, psnr, ss = eval_model(ck, files)
        print(f"{name:<26} {n:>3}  {l1*4:>7.4f}  {l1:>7.4f}  {psnr:>6.2f}  {ss:>6.4f}")


if __name__ == "__main__":
    main()
