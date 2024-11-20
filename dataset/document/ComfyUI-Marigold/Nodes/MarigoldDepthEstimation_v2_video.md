# MarigoldDepthEstimation_v2_video
## Documentation
- Class name: `MarigoldDepthEstimation_v2_video`
- Category: `Marigold`
- Output node: `False`

The MarigoldDepthEstimation_v2_video node is designed for depth estimation in videos using a diffusion-based approach. It incorporates optical flow to ensure consistency between video frames, enhancing the depth estimation process by leveraging temporal information. This node is an experimental version that aims to provide more accurate and visually coherent depth maps for video sequences.
## Input types
### Required
- **`marigold_model`**
    - Specifies the Marigold model to be used for depth estimation. This selection influences the accuracy and quality of the depth maps generated.
    - Comfy dtype: `MARIGOLDMODEL`
    - Python dtype: `str`
- **`images`**
    - A collection of images or a single image to be processed for depth estimation. The images serve as input for generating depth maps.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image] or Image`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the depth estimation results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise_steps`**
    - The number of denoising steps to apply, affecting the clarity and detail of the generated depth maps.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`processing_resolution`**
    - The resolution at which the images are processed, impacting the balance between processing speed and depth map quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scheduler`**
    - The scheduler type used for controlling the diffusion process, affecting the characteristics of the depth estimation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`blend_factor`**
    - Determines the blending ratio of the current frame with the previous frame's latent representation, enhancing temporal consistency in videos.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_taesd_vae`**
    - A boolean flag indicating whether to use the TAESD VAE model for improved depth estimation accuracy.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after depth estimation, representing the depth map of the input image(s).
    - Python dtype: `Image`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MarigoldDepthEstimation_v2_video:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "marigold_model": ("MARIGOLDMODEL",),  
            "images": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            "processing_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
            "scheduler": (
            ["DDIMScheduler", "LCMScheduler",], 
            {
               "default": 'LCMScheduler'
            }),
            
            "blend_factor": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 1.0, "step": 0.01}),
            "use_taesd_vae": ("BOOLEAN", {"default": True}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =("image",)
    FUNCTION = "process"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
Uses Diffusers 0.28.0 Marigold pipelines.  
This node uses the previous frame as init latent to  
smooth out the video.  
"""

    def process(self, marigold_model, images, seed, denoise_steps, processing_resolution, blend_factor, scheduler, use_taesd_vae):
        try:
            from diffusers import AutoencoderTiny
        except:
            raise Exception("diffusers==0.28 is required for v2 nodes")
        device = model_management.get_torch_device()
        
        pipeline = marigold_model['pipeline']
        pred_type = marigold_model['modeltype']

        if use_taesd_vae:
            pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device)

        scheduler_kwargs = {
            DDIMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": 1,
            },
            LCMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": 1,
            },	
        }
        if scheduler == 'DDIMScheduler':
            pipe_kwargs = scheduler_kwargs[DDIMScheduler]
        elif scheduler == 'LCMScheduler':
            pipe_kwargs = scheduler_kwargs[LCMScheduler]

        
        B, H, W, C  = images.shape
        size = [W, H]
        images = images.permute(0, 3, 1, 2).to(device)

        last_frame_latent = None
        torch.manual_seed(seed)
        latent_common = torch.randn((1, 4, processing_resolution * size[1] // (8 * max(size)), processing_resolution * size[0] // (8 * max(size)))).to(device=device, dtype=torch.float16)
        pbar = comfy.utils.ProgressBar(B)
        processed_out = []
        for img in images:
            latents = latent_common
            if last_frame_latent is not None:
                latents = (1 - blend_factor) * latents + blend_factor * last_frame_latent

            processed = pipeline(
                img,
                processing_resolution = processing_resolution,
                match_input_resolution=False, 
                latents=latents,
                output_latent=True,
                output_type = "pt",
                **pipe_kwargs
                )
            last_frame_latent = processed.latent
            pbar.update(1)
            if pred_type == "normals":
                normals = pipeline.image_processor.visualize_normals(processed.prediction)
                normals_tensor = transforms.ToTensor()(normals[0])
                processed_out.append(normals_tensor)
            else:
                processed_out.append(processed[0])
        
        if pred_type == "normals":
            processed_out = torch.stack(processed_out, dim=0)
            processed_out = processed_out.permute(0, 2, 3, 1).cpu().float()
        else:
            processed_out = torch.cat(processed_out, dim=0)
            processed_out = processed_out.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).cpu().float()
            processed_out = 1.0 - processed_out

        return (processed_out,)

```
