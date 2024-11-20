---
tags:
- Image
---

# MarigoldDepthEstimation_v2
## Documentation
- Class name: `MarigoldDepthEstimation_v2`
- Category: `Marigold`
- Output node: `False`

The MarigoldDepthEstimation_v2 node is designed for diffusion-based monocular depth estimation, leveraging the Marigold model to generate depth maps from single images. This node focuses on enhancing the accuracy of depth predictions through iterative processing and ensemble techniques, offering customizable parameters for fine-tuning the depth estimation process.
## Input types
### Required
- **`marigold_model`**
    - Selects the specific Marigold model to be used for depth estimation, affecting the depth map's accuracy and appearance.
    - Comfy dtype: `MARIGOLDMODEL`
    - Python dtype: `str`
- **`image`**
    - The input image for which the depth map is to be estimated, serving as the basis for the depth estimation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`seed`**
    - Sets the seed for random number generation, ensuring reproducibility of the depth estimation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise_steps`**
    - Specifies the number of steps per depth map, allowing users to balance between accuracy and processing time.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`ensemble_size`**
    - Determines the number of depth maps to be ensembled into a single output, enhancing the depth estimation's accuracy.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`processing_resolution`**
    - Defines the resolution at which the depth estimation process is performed, impacting the detail and quality of the depth map.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scheduler`**
    - Chooses the scheduler for the depth estimation process, influencing the final depth map's characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_taesd_vae`**
    - Determines whether to use the TAESD VAE model for depth estimation, potentially improving the quality of the depth map.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Produces a depth map image, representing the estimated depth of the scene from a single input image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MarigoldDepthEstimation_v2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "marigold_model": ("MARIGOLDMODEL",),
            "image": ("IMAGE", ),
            "seed": ("INT", {"default": 123,"min": 0, "max": 0xffffffffffffffff, "step": 1}),
            "denoise_steps": ("INT", {"default": 4, "min": 1, "max": 4096, "step": 1}),
            "ensemble_size": ("INT", {"default": 3, "min": 1, "max": 4096, "step": 1}),
            "processing_resolution": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
            "scheduler": (
            ["DDIMScheduler", "LCMScheduler",], 
            {
               "default": 'LCMScheduler'
            }),
            "use_taesd_vae": ("BOOLEAN", {"default": False}),
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
"""

    def process(self, marigold_model, image, seed, denoise_steps, processing_resolution, ensemble_size, scheduler, use_taesd_vae):
        try:
            from diffusers import AutoencoderTiny
        except:
            raise Exception("diffusers==0.28 is required for v2 nodes")
        batch_size = image.shape[0]
        device = model_management.get_torch_device()
        torch.manual_seed(seed)

        image = image.permute(0, 3, 1, 2).to(device)

        pipeline = marigold_model['pipeline']
        pred_type = marigold_model['modeltype']

        if use_taesd_vae:
            pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device)
            
        pbar = comfy.utils.ProgressBar(batch_size)

        scheduler_kwargs = {
            DDIMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": ensemble_size,
            },
            LCMScheduler: {
                "num_inference_steps": denoise_steps,
                "ensemble_size": ensemble_size,
            },	
        }
        if scheduler == 'DDIMScheduler':
            pipe_kwargs = scheduler_kwargs[DDIMScheduler]
        elif scheduler == 'LCMScheduler':
            pipe_kwargs = scheduler_kwargs[LCMScheduler]

        generator = torch.Generator(device).manual_seed(seed)

        processed_out = []

        for i in range(batch_size):
            processed = pipeline(
                image[i],
                output_type = "pt",
                generator = generator,
                processing_resolution = processing_resolution,
                **pipe_kwargs
                )
            
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
