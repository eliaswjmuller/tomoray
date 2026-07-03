import os
import numpy as np
from pathlib import Path
import torch
import lpips
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from einops import rearrange
from torchvision import transforms as T
from PIL import Image


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

class Metrics:
    def __init__(self, path, val_dl, device=None):

        self.val_dl = val_dl
        self.root_dir = path
        self.path_to_save = os.path.join(self.root_dir, "metrics")
        self.path_to_save_PIPS = os.path.join(self.path_to_save, "PIPS")
        self.path_to_save_SSIM = os.path.join(self.path_to_save, "SSIM")
        self.path_to_save_PSNR = os.path.join(self.path_to_save, "PSNR")
        self.path_to_save_RMSE = os.path.join(self.path_to_save, "RMSE")
        self.path_to_save_FID = os.path.join(self.path_to_save, "FID")

        Path(self.path_to_save_SSIM).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PIPS).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_PSNR).mkdir(parents=True, exist_ok=True)
        Path(self.path_to_save_RMSE).mkdir(parents=True, exist_ok=True)

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        self.loss_lpips = lpips.LPIPS(net='vgg').to(self.device)


    def process_input(self, data):

        img = data["image"].to(self.device).float()
        xrays = data["projections"].to(self.device).float()
        angles = data["angles"].to(self.device).float()

        return img, xrays, angles



    def sample(self, model):
        model.eval()
        
        with torch.no_grad():
            return
    

    def compute_metrics(self, model, fusion_model):

        if self.val_dl is None:
            print('No validation set')
            return 0.0, 0.0, 0.0, 0.0

        pips, ssim, psnr, rmse, fid = [], [], [], [], []
        model.eval()
        fusion_model.eval()
        N = len(self.val_dl)
            
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                ct, xrays, angles = self.process_input(batch)

                cond = fusion_model(xrays, angles)
                
                #gen = model.sample(cond=cond, batch_size=ct.shape[0])
                gen = model.sample_dpm(cond=cond, batch_size=ct.shape[0], steps=20)

                #input1 = (gen + 1) / 2
                #input2 = (ct + 1) / 2

                input1 = torch.clamp((gen + 1) / 2, 0, 1)
                input2 = torch.clamp((ct + 1) / 2, 0, 1)

                ssim_value, _ = self.ssim_3d(input1, input2)
                ssim.append(ssim_value.item() if isinstance(ssim_value, torch.Tensor) else ssim_value)

                pips_value = self.pips_3d(input1, input2)
                pips.append(pips_value.item() if isinstance(pips_value, torch.Tensor) else pips_value)

                psnr_value = self.psnr_3d(input1, input2)
                psnr.append(psnr_value.item() if isinstance(psnr_value, torch.Tensor) else psnr_value)

                mse = torch.mean((input1 - input2) ** 2)
                rmse_value = torch.sqrt(mse)
                rmse.append(rmse_value.item())

        model.train()
        fusion_model.train()

        avg_pips = sum(pips) / len(pips) if pips else 0
        avg_ssim = sum(ssim) / len(ssim) if ssim else 0
        avg_psnr = sum(psnr) / len(psnr) if psnr else 0
        avg_rmse = sum(rmse) / len(rmse) if rmse else 0

        return avg_pips, avg_ssim, avg_psnr, avg_rmse
    


    def compute_metrics_test(self, model, fusion_model, save_slice_image=True, save_nifti=True, save_gif=True):
        pips, ssim, psnr, rmse, fid = [], [], [], [], []
        model.eval()
        fusion_model.eval()

        path_results = os.path.join(self.root_dir, "test_results")
        os.makedirs(path_results, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                ct, xrays, angles = self.process_input(batch)
                cond = fusion_model(xrays, angles)

                #gen = model.sample(cond=cond, batch_size= ct.shape[0])
                gen = model.sample_dpm(cond=cond, batch_size=ct.shape[0], steps=20)

                input1 = (gen + 1) / 2
                input2 = (ct + 1) / 2

                generated_np = input1.cpu().numpy()
                image_np = input2.cpu().numpy()

                xray_vis = xrays[:, :, 0, :, :].cpu().numpy()

                if save_nifti:
                    self._save_nifti(generated_np, image_np, path_results, i)
                
                if save_slice_image:
                    self._save_slices(generated_np, image_np, xray_vis, path_results, i)

                if save_gif:
                    all_gen_list = F.pad(gen, (2, 2, 2, 2)) 
                    all_real_list = F.pad(ct, (2, 2, 2, 2))  # real_ct (image)
                    

                    all_xray_list = F.pad(xrays, (2, 2, 2, 2))


                    gen_gif_tensor = rearrange(all_gen_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                    real_gif_tensor = rearrange(all_real_list, '(i j) c f h w -> c f (i h) (j w)', i=1)
                    xray_gif_tensor = rearrange(all_xray_list, '(i j) c f h w -> c f (i h) (j w)', i=1)

                    path_video = os.path.join(self.root_dir, 'video_results')
                    os.makedirs(path_video, exist_ok=True)


                    video_tensor_to_gif(gen_gif_tensor, os.path.join(path_video, f'{i}_generated.gif'))
                    video_tensor_to_gif(real_gif_tensor, os.path.join(path_video, f'{i}_real.gif'))
                    video_tensor_to_gif(xray_gif_tensor, os.path.join(path_video, f'{i}_xray_input.gif'))

                    if i > 50:
                        save_gif = False


                ssim_value, _ = self.ssim_3d(input1, input2)
                ssim.append(ssim_value.item() if isinstance(ssim_value, torch.Tensor) else ssim_value)

                pips_value = self.pips_3d(input1, input2)
                pips.append(pips_value.item() if isinstance(pips_value, torch.Tensor) else pips_value)

                psnr_value = self.psnr_3d(input1, input2)
                psnr.append(psnr_value.item() if isinstance(psnr_value, torch.Tensor) else psnr_value)

                mse = torch.mean((input1 - input2) ** 2)
                rmse_value = torch.sqrt(mse)
                rmse.append(rmse_value.item())

        avg_pips = sum(pips) / len(pips) if pips else 0
        avg_ssim = sum(ssim) / len(ssim) if ssim else 0
        avg_psnr = sum(psnr) / len(psnr) if psnr else 0
        avg_rmse = sum(rmse) / len(rmse) if rmse else 0

        return avg_pips, avg_ssim, avg_psnr, avg_rmse
    


    def _save_nifti(self, gen_np, real_np, root, idx):
        path_fake = os.path.join(root, 'nifti', 'fake')
        path_real = os.path.join(root, 'nifti', 'real')
        os.makedirs(path_fake, exist_ok=True)
        os.makedirs(path_real, exist_ok=True)
        
        vol_gen = np.transpose(gen_np[0, 0], (1, 2, 0))
        vol_real = np.transpose(real_np[0, 0], (1, 2, 0))
        
        nib.save(nib.Nifti1Image(vol_gen, np.eye(4)), os.path.join(path_fake, f'{idx}_gen.nii.gz'))
        nib.save(nib.Nifti1Image(vol_real, np.eye(4)), os.path.join(path_real, f'{idx}_real.nii.gz'))

    def _save_slices(self, gen_np, real_np, xray_np, root, idx):
        path_slices = os.path.join(root, 'slices')
        os.makedirs(path_slices, exist_ok=True)
        
        mid_slice = gen_np.shape[2] // 2
        

        xray_img = xray_np[0, 0]
        gen_img = gen_np[0, 0, mid_slice]
        real_img = real_np[0, 0, mid_slice]
        
        def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-8)
        
        combined = np.concatenate((norm(xray_img), norm(gen_img), norm(real_img)), axis=1)
        
        plt.figure(figsize=(12, 4))
        plt.imshow(combined, cmap='gray')
        plt.title(f"X-Ray Input (View 0)  |  Generated CT  |  Real CT")
        plt.axis('off')
        plt.savefig(os.path.join(path_slices, f'{idx}_comparison.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_gif(self, real_vol, fake_vol, milestone, idx=0, phase="val"):
        """
        real_vol, fake_vol : (D, H, W) ou (1, D, H, W)
        xray : (H, W) ou (1, H, W)
        """
        path_video = os.path.join(self.root_dir, 'video_results', phase)
        os.makedirs(path_video, exist_ok=True)
        
        if real_vol.ndim == 4: real_vol = real_vol[0]
        if fake_vol.ndim == 4: fake_vol = fake_vol[0]

        def to_uint8(t):
            t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            return (t * 255).astype(np.uint8)

        real_uint = to_uint8(real_vol) # (D, H, W)
        fake_uint = to_uint8(fake_vol) # (D, H, W)  
        

        frames_axial = []
        for d in range(real_uint.shape[0]):
            frame = np.concatenate((fake_uint[d, :, :], real_uint[d, :, :]), axis=1)
            frames_axial.append(Image.fromarray(frame))
            
        frames_axial[0].save(
            os.path.join(path_video, f'step_{milestone}_sample_{idx}_1_axial.gif'), 
            save_all=True, append_images=frames_axial[1:], duration=100, loop=0
        )
        if phase == "test":

            frames_coronal = []
            for h in range(real_uint.shape[1]):
                frame = np.concatenate((fake_uint[:, h, :], real_uint[:, h, :]), axis=1)
                frames_coronal.append(Image.fromarray(frame))
                
            frames_coronal[0].save(
                os.path.join(path_video, f'step_{milestone}_sample_{idx}_2_coronal.gif'), 
                save_all=True, append_images=frames_coronal[1:], duration=100, loop=0
            )

            frames_sagittal = []
            for w in range(real_uint.shape[2]):
                frame = np.concatenate((fake_uint[:, :, w], real_uint[:, :, w]), axis=1)
                frames_sagittal.append(Image.fromarray(frame))
                
            frames_sagittal[0].save(
                os.path.join(path_video, f'step_{milestone}_sample_{idx}_3_sagittal.gif'), 
                save_all=True, append_images=frames_sagittal[1:], duration=100, loop=0
            )

    def pips_3d(self, img1, img2):
        # img: (B, C, D, H, W)
        assert img1.shape == img2.shape
        b, c, d, h, w = img1.shape
        total_loss = 0.0

        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        for i in range(d):
            sl1 = img1[:, :, i, :, :]
            sl2 = img2[:, :, i, :, :]
            
            sl1 = sl1 * 2 - 1
            sl2 = sl2 * 2 - 1
                
            total_loss += self.loss_lpips(sl1, sl2)
            
        return total_loss / d

    def psnr_3d(self, img1, img2):
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr
    
    def ssim_3d(self, img1, img2, window_size=11, size_average=True, val_range=None):
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        if val_range is None:   
            max_val = 255 if torch.max(img1) > 128 else 1
            min_val = -1 if torch.min(img1) < -0.5 else 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, depth, height, width) = img1.size()
        window = create_window(window_size, channel=channel).to(img1.device)

        mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv3d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            return ssim_map.mean(), cs
        else:
            return ssim_map.mean(1).mean(1).mean(1), cs

    def update_metrics(self, model, fusion_model, cur_iter):
        print("--- Iter %s: computing PIPS SSIM PSNR RMSE ---" % (cur_iter))
        cur_pips, cur_ssim, cur_psnr, cur_rmse = self.compute_metrics(model, fusion_model)
        self.update_logs(cur_pips, cur_iter, 'PIPS')
        self.update_logs(cur_ssim, cur_iter, 'SSIM')
        self.update_logs(cur_psnr, cur_iter, 'PSNR')
        self.update_logs(cur_rmse, cur_iter, 'RMSE')
        print("--- Metrics at Iter %s: PIPS: %.2f | SSIM: %.2f | PSNR: %.2f | RMSE: %.2f" % (cur_iter, cur_pips, cur_ssim, cur_psnr, cur_rmse))

    def update_logs(self, cur_data, epoch, mode):
        try:
            os.makedirs(os.path.join(self.path_to_save, mode), exist_ok=True)
            log_path = os.path.join(self.path_to_save, mode, f"{mode}_log.npy")
            if os.path.exists(log_path):
                np_file = np.load(log_path, allow_pickle=True)
                first = list(np_file[0, :])
                sercon = list(np_file[1, :])
            else:
                first = []
                sercon = []
            
            first.append(epoch)
            sercon.append(cur_data)
            np_file = [first, sercon]
        except Exception as e:
            print(f"Logging error: {e}")
            np_file = [[epoch], [cur_data]]

        np.save(os.path.join(self.path_to_save, mode, f"{mode}_log.npy"), np_file)
        np_file = np.array(np_file)
        plt.figure()
        plt.plot(np_file[0, :], np_file[1, :])
        plt.grid(visible=True, which='major', color='#666666', linestyle='--')
        plt.minorticks_on()
        plt.grid(visible=True, which='minor', color='#999999', linestyle='--', alpha=0.2)
        plt.title(f"{mode} over iterations")
        plt.savefig(os.path.join(self.path_to_save, mode, f"plot_{mode}.png"), dpi=300)
        plt.close()

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())) * 1.0
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   duration=duration, loop=loop, optimize=optimize)
    return images