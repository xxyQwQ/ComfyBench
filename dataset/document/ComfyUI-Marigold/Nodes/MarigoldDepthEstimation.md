---
tags:
- DepthMap
- DepthMapEstimation
- Image
---

# MarigoldDepthEstimation
## Documentation
- Class name: `MarigoldDepthEstimation`
- Category: `Marigold`
- Output node: `False`

The MarigoldDepthEstimation node is designed for diffusion-based monocular depth estimation, leveraging the Marigold model to generate depth maps from single images. This node focuses on enhancing the accuracy of depth perception in images by employing a series of denoising steps and iterations, allowing for the creation of ensembled depth maps that offer a more detailed and accurate representation of depth. It supports various configurations, including model selection and inversion of depth map colors, to cater to different use cases and preferences.
## Input types
### Required
- **`image`**
    - The input image for which the depth map is to be estimated. This is the primary input over which the depth estimation process is executed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the depth estimation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise_steps`**
    - Specifies the number of steps per depth map to enhance the accuracy of the depth estimation. Increasing this value results in more detailed depth maps at the cost of longer processing times.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`n_repeat`**
    - Determines the amount of iterations to be ensembled into a single depth map, affecting the overall depth accuracy and detail.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`regularizer_strength`**
    - Adjusts the strength of the regularization during the ensembling process, typically left at default settings.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`reduction_method`**
    - Specifies the method used for reducing the ensemble of depth maps into a single map, impacting the final depth representation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_iter`**
    - Sets the maximum number of iterations for the ensembling process, controlling the depth map's refinement level.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tol`**
    - Defines the tolerance level for the ensembling process, determining when the process should terminate based on convergence criteria.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`invert`**
    - Inverts the default depth map colors from black representing the front to the opposite, catering to specific application requirements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`keep_model_loaded`**
    - Indicates whether the Marigold model should remain loaded between invocations, improving performance at the expense of memory usage.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`n_repeat_batch_size`**
    - Controls how many of the n_repeat iterations are processed in a batch, optimizing VRAM usage and processing speed based on available resources.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`use_fp16`**
    - Determines whether to use fp16 precision for processing, affecting VRAM usage and potentially the depth map's quality.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`scheduler`**
    - Chooses the scheduler for the denoising steps, affecting the depth map's final quality and characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`normalize`**
    - Specifies whether the depth map should be normalized, affecting the range and distribution of depth values.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`model`**
    - Selects between the Marigold model and its LCM version for depth estimation, influencing the depth map's accuracy and appearance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`ensembled_image`**
    - Comfy dtype: `IMAGE`
    - The resulting depth map, ensembled from multiple iterations and denoising steps, providing a detailed representation of depth in the image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MarigoldDepthEstimation:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {  
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "n_repeat": ("INT", {"default": 10, "min": 1, "max": 4096, "step": 1}),
            "regularizer_strength": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 4096, "step": 0.001}),
            "reduction_method": (
            [   
                'median',
                'mean',  
            ], {
               "default": 'median'
            }),
            "max_iter": ("INT", {"default": 5, "min": 1, "max": 4096, "step": 1}),
            "tol": ("FLOAT", {"default": 1e-3, "min": 1e-6, "max": 1e-1, "step": 1e-6}),
            
            "invert": ("BOOLEAN", {"default": True}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
            "n_repeat_batch_size": ("INT", {"default": 2, "min": 1, "max": 4096, "step": 1}),
            "use_fp16": ("BOOLEAN", {"default": True}),
            "scheduler": (
            [   
                'DDIMScheduler',
                'DDPMScheduler',
                'PNDMScheduler',
                'DEISMultistepScheduler',
                'LCMScheduler',
            ], {
               "default": 'DDIMScheduler'
            }),
            "normalize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "model": (
            [   
                'Marigold',
                'marigold-lcm-v1-0',  
            ], {
               "default": 'Marigold'
            }),
            }
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("ensembled_image",)
    FUNCTION = "process"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
- denoise_steps: steps per depth map, increase for accuracy in exchange of processing time
- n_repeat: amount of iterations to be ensembled into single depth map
- n_repeat_batch_size: how many of the n_repeats are processed as a batch,  
if you have the VRAM this can match the n_repeats for faster processing  
- model: Marigold or it's LCM version marigold-lcm-v1-0  
For the LCM model use around 4 steps and the LCMScheduler  
- scheduler: Different schedulers give bit different results  
- invert: marigold by default produces depth map where black is front,  
for controlnets etc. we want the opposite.  
- regularizer_strength, reduction_method, max_iter, tol (tolerance) are settings   
for the ensembling process, generally do not touch.  
- use_fp16: if true, use fp16, if false use fp32  
fp16 uses much less VRAM, but in some cases can lead to loss of quality.  
"""

    def process(self, image, seed, denoise_steps, n_repeat, regularizer_strength, reduction_method, max_iter, tol,invert, keep_model_loaded, n_repeat_batch_size, use_fp16, scheduler, normalize, model="Marigold"):
        batch_size = image.shape[0]
        precision = torch.float16 if use_fp16 else torch.float32
        device = model_management.get_torch_device()
        torch.manual_seed(seed)

        image = image.permute(0, 3, 1, 2).to(device).to(dtype=precision)
        if normalize:
            image = image * 2.0 - 1.0

        diffusers_model_path = os.path.join(folder_paths.models_dir,'diffusers')
        #load the diffusers model
        if model == "Marigold":
            folders_to_check = [
                os.path.join(script_directory,"checkpoints","Marigold_v1_merged",),
                os.path.join(script_directory,"checkpoints","Marigold",),
                os.path.join(diffusers_model_path,"Marigold_v1_merged"),
                os.path.join(diffusers_model_path,"Marigold")
            ]
        elif model == "marigold-lcm-v1-0":
            folders_to_check = [
                os.path.join(diffusers_model_path,"marigold-lcm-v1-0"),
                os.path.join(diffusers_model_path,"checkpoints","marigold-lcm-v1-0")
            ]
        self.custom_config = {
            "model": model,
            "use_fp16": use_fp16,
            "scheduler": scheduler,
        }
        if not hasattr(self, 'marigold_pipeline') or self.marigold_pipeline is None or self.current_config != self.custom_config:
            self.current_config = self.custom_config
            # Load the model only if it hasn't been loaded before
            checkpoint_path = None
            for folder in folders_to_check:
                if os.path.exists(folder):
                    checkpoint_path = folder
                    break
            to_ignore = ["*.bin", "*fp16*"]

            if checkpoint_path is None:
                if model == "Marigold":
                    try:
                        from huggingface_hub import snapshot_download
                        checkpoint_path = os.path.join(diffusers_model_path, "Marigold")
                        snapshot_download(repo_id="Bingxin/Marigold", ignore_patterns=to_ignore, local_dir=checkpoint_path, local_dir_use_symlinks=False)  
                    except:
                        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_path}")
                if model == "marigold-lcm-v1-0":
                    try:
                        from huggingface_hub import snapshot_download
                        checkpoint_path = os.path.join(diffusers_model_path, "marigold-lcm-v1-0")
                        snapshot_download(repo_id="prs-eth/marigold-lcm-v1-0", ignore_patterns=to_ignore, local_dir=checkpoint_path, local_dir_use_symlinks=False)  
                    except:
                        raise FileNotFoundError(f"No checkpoint directory found at {checkpoint_path}")

            self.marigold_pipeline = MarigoldPipeline.from_pretrained(checkpoint_path, enable_xformers=False, empty_text_embed=empty_text_embed, noise_scheduler_type=scheduler)
            self.marigold_pipeline = self.marigold_pipeline.to(device).half() if use_fp16 else self.marigold_pipeline.to(device)
            self.marigold_pipeline.unet.eval()  # Set the model to evaluation mode
        pbar = comfy.utils.ProgressBar(batch_size * n_repeat)

        out = []

        with torch.no_grad():
            for i in range(batch_size):
                # Duplicate the current image n_repeat times
                duplicated_batch = image[i].unsqueeze(0).repeat(n_repeat, 1, 1, 1)
                
                # Process the duplicated batch in sub-batches
                depth_maps = []
                for j in range(0, n_repeat, n_repeat_batch_size):
                    # Get the current sub-batch
                    sub_batch = duplicated_batch[j:j + n_repeat_batch_size]
                    
                    # Process the sub-batch
                    depth_maps_sub_batch = self.marigold_pipeline(sub_batch, num_inference_steps=denoise_steps, show_pbar=False)
                    
                    # Process each depth map in the sub-batch if necessary
                    for depth_map in depth_maps_sub_batch:
                        depth_map = torch.clip(depth_map, -1.0, 1.0)
                        depth_map = (depth_map + 1.0) / 2.0
                        depth_maps.append(depth_map)
                        pbar.update(1)
                
                depth_predictions = torch.cat(depth_maps, dim=0).squeeze()
                del duplicated_batch, depth_maps_sub_batch
                torch.cuda.empty_cache()  # clear vram cache for ensembling

                # Test-time ensembling
                if n_repeat > 1:
                    depth_map, pred_uncert = ensemble_depths(
                        depth_predictions,
                        regularizer_strength=regularizer_strength,
                        max_iter=max_iter,
                        tol=tol,
                        reduction=reduction_method,
                        max_res=None,
                        device=device,
                    )
                    print(depth_map.shape)
                    depth_map = depth_map.unsqueeze(2).repeat(1, 1, 3)
                    print(depth_map.shape)
                else:
                    depth_map = depth_map.permute(1, 2, 0)
                    depth_map = depth_map.repeat(1, 1, 3)
                    print(depth_map.shape)
                
                out.append(depth_map)
                del depth_map, depth_predictions
      
        if invert:
            outstack = 1.0 - torch.stack(out, dim=0).cpu().to(torch.float32)
        else:
            outstack = torch.stack(out, dim=0).cpu().to(torch.float32)

        if not keep_model_loaded:
            self.marigold_pipeline = None
            model_management.soft_empty_cache()
        return (outstack,)        

```
