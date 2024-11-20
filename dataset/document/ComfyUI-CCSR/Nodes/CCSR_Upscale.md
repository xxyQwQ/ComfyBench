---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# CCSR_Upscale
## Documentation
- Class name: `CCSR_Upscale`
- Category: `CCSR`
- Output node: `False`

The node is designed to upscale images or latent representations, leveraging custom scaling methods to enhance the resolution or detail of the input data. It focuses on applying advanced upscaling techniques to improve image quality or adjust the size of latent representations for further processing or visualization.
## Input types
### Required
- **`ccsr_model`**
    - Specifies the CCSR model to be used for upscaling, which determines the algorithm and approach for enhancing the image or latent representation.
    - Comfy dtype: `CCSRMODEL`
    - Python dtype: `CCSRModel`
- **`image`**
    - Represents the input image to be upscaled. The quality and characteristics of the input image directly influence the outcome of the upscaling process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`resize_method`**
    - Determines the method to be used for resizing during the upscaling process, offering a variety of algorithms such as lanczos among others.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale_by`**
    - Defines the factor by which the image or latent representation is to be scaled. This value allows for precise control over the enlargement of the input data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Specifies the number of steps to be used in the upscaling process, affecting the detail and quality of the upscaled output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`t_max`**
    - The maximum threshold for the upscaling process, influencing the extent of detail enhancement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`t_min`**
    - The minimum threshold for the upscaling process, affecting the base level of detail to be enhanced.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampling_method`**
    - Specifies the sampling method to be used in the upscaling process, affecting the algorithm's approach to enhancing detail and resolution.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tile_size`**
    - Defines the size of tiles to be used in the upscaling process, allowing for efficient processing of large images by breaking them into manageable pieces.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`tile_stride`**
    - Determines the stride of tiling during the upscaling process, affecting the overlap and integration of upscaled tiles.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`vae_tile_size_encode`**
    - Specifies the tile size for the encoding phase in the upscaling process, affecting the resolution and detail of the encoded representation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`vae_tile_size_decode`**
    - Specifies the tile size for the decoding phase in the upscaling process, affecting the resolution and detail of the decoded output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color_fix_type`**
    - Determines the method for color correction during the upscaling process, offering options like 'adain' and 'wavelet' for enhanced visual fidelity.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`keep_model_loaded`**
    - Indicates whether the CCSR model should remain loaded in memory after the upscaling process, affecting resource utilization and performance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`seed`**
    - Specifies the seed value for random number generation during the upscaling process, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`upscaled_image`**
    - Comfy dtype: `IMAGE`
    - The result of the upscaling process, which is an enhanced-resolution version of the input samples. This output reflects the applied upscaling method and target dimensions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CCSR_Upscale:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ccsr_model": ("CCSRMODEL", ),
            "image": ("IMAGE", ),
            "resize_method": (s.upscale_methods, {"default": "lanczos"}),
            "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01}),
            "steps": ("INT", {"default": 45, "min": 3, "max": 4096, "step": 1}),
            "t_max": ("FLOAT", {"default": 0.6667,"min": 0, "max": 1, "step": 0.01}),
            "t_min": ("FLOAT", {"default": 0.3333,"min": 0, "max": 1, "step": 0.01}),
            "sampling_method": (
            [   
                'ccsr',
                'ccsr_tiled_mixdiff',
                'ccsr_tiled_vae_gaussian_weights',
            ], {
               "default": 'ccsr_tiled_mixdiff'
            }),
            "tile_size": ("INT", {"default": 512, "min": 1, "max": 4096, "step": 1}),
            "tile_stride": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 1}),
            "vae_tile_size_encode": ("INT", {"default": 1024, "min": 2, "max": 4096, "step": 8}),
            "vae_tile_size_decode": ("INT", {"default": 1024, "min": 2, "max": 4096, "step": 8}),
            "color_fix_type": (
            [   
                'none',
                'adain',
                'wavelet',
            ], {
               "default": 'adain'
            }),
            "keep_model_loaded": ("BOOLEAN", {"default": False}),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            },
            
            
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("upscaled_image",)
    FUNCTION = "process"

    CATEGORY = "CCSR"

    @torch.no_grad()
    def process(self, ccsr_model, image, resize_method, scale_by, steps, t_max, t_min, tile_size, tile_stride, 
                color_fix_type, keep_model_loaded, vae_tile_size_encode, vae_tile_size_decode, sampling_method, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mm.unload_all_models()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = ccsr_model['dtype']
        model = ccsr_model['model']
        
        #empty_text_embed = torch.load(os.path.join(script_directory, "empty_text_embed.pt"), map_location=device)
        empty_text_embed_sd = comfy.utils.load_torch_file(os.path.join(script_directory, "empty_text_embed.safetensors"))
        empty_text_embed = empty_text_embed_sd['empty_text_embed'].to(dtype).to(device)

        sampler = SpacedSampler(model, var_type="fixed_small")

        image, = ImageScaleBy.upscale(self, image, resize_method, scale_by)
        
        B, H, W, C = image.shape

        # Calculate the new height and width, rounding down to the nearest multiple of 64.
        new_height = H // 64 * 64
        new_width = W // 64 * 64

        # Reorder to [B, C, H, W] before using interpolate.
        image = image.permute(0, 3, 1, 2).contiguous()
        resized_image = F.interpolate(image, size=(new_height, new_width), mode='bilinear', align_corners=False)
 
        strength = 1.0
        model.control_scales = [strength] * 13
        
        model.to(device, dtype=dtype).eval()

        height, width = resized_image.size(-2), resized_image.size(-1)
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)

        out = []
        if B > 1:
            pbar = comfy.utils.ProgressBar(B)
        autocast_condition = dtype == torch.float16 and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for i in range(B):
                img = resized_image[i].unsqueeze(0).to(device)
                if sampling_method == 'ccsr_tiled_mixdiff':
                    model.reset_encoder_decoder()
                    print("Using tiled mixdiff")
                    samples = sampler.sample_with_mixdiff_ccsr(
                        empty_text_embed, tile_size=tile_size, tile_stride=tile_stride,
                        steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
                        positive_prompt="", negative_prompt="", x_T=x_T,
                        cfg_scale=1.0, 
                        color_fix_type=color_fix_type
                    )
                elif sampling_method == 'ccsr_tiled_vae_gaussian_weights':
                    model._init_tiled_vae(encoder_tile_size=vae_tile_size_encode // 8, decoder_tile_size=vae_tile_size_decode // 8)
                    print("Using gaussian weights")
                    samples = sampler.sample_with_tile_ccsr(
                        empty_text_embed, tile_size=tile_size, tile_stride=tile_stride,
                        steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
                        positive_prompt="", negative_prompt="", x_T=x_T,
                        cfg_scale=1.0, 
                        color_fix_type=color_fix_type
                    )
                else:
                    model.reset_encoder_decoder()
                    print("no tiling")
                    samples = sampler.sample_ccsr(
                        empty_text_embed, steps=steps, t_max=t_max, t_min=t_min, shape=shape, cond_img=img,
                        positive_prompt="", negative_prompt="", x_T=x_T,
                        cfg_scale=1.0,
                        color_fix_type=color_fix_type
                    )
                out.append(samples.squeeze(0).cpu())
                mm.throw_exception_if_processing_interrupted()
                if B > 1:
                    pbar.update(1)
                    print("Sampled image ", i, " out of ", B)
       
        original_height, original_width = H, W  
        processed_height = samples.size(2)
        target_width = int(processed_height * (original_width / original_height))
        out_stacked = torch.stack(out, dim=0).cpu().to(torch.float32).permute(0, 2, 3, 1)
        resized_back_image, = ImageScale.upscale(self, out_stacked, "lanczos", target_width, processed_height, crop="disabled")
        
        if not keep_model_loaded:
            model.to(offload_device)           
            mm.soft_empty_cache()
        return(resized_back_image,)

```
