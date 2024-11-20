---
tags:
- LoRA
---

# Intrinsic Lora Sampling
## Documentation
- Class name: `Intrinsic_lora_sampling`
- Category: `KJNodes`
- Output node: `False`

This node enables the application of intrinsic LoRAs (Low-Rank Adaptations) to models for specific tasks such as depth map generation, surface normal estimation, albedo, and shading. It leverages LoRAs to modify the behavior of pre-trained models without extensive retraining, focusing on enhancing or altering the model's output based on the task at hand.
## Input types
### Required
- **`model`**
    - The model to which the intrinsic LoRA will be applied. It serves as the base for modifications and enhancements specific to the desired task.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`lora_name`**
    - The name of the LoRA to be applied, selected from a predefined list of available intrinsic LoRAs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`task`**
    - Specifies the task for which the model's output is being adapted, such as depth map generation or surface normal estimation. The default task is depth map generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text`**
    - A text prompt that guides the model's adaptation process for the specified task, enhancing the relevance of the output to the input text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A CLIP model used for encoding the text prompt into a format that guides the adaptation process.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - A VAE (Variational Autoencoder) model used for encoding or decoding the samples during the adaptation process.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`per_batch`**
    - The number of samples processed per batch, affecting the efficiency and speed of the adaptation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image`**
    - An optional input image that can be used as a basis for the adaptation process, providing a visual context.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`optional_latent`**
    - An optional latent representation that can be used instead of generating one from an input image, offering a shortcut in the adaptation process.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the intrinsic LoRA, adapted for the specified task.
    - Python dtype: `torch.Tensor`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation of the output image, providing a deeper insight into the model's adaptation process.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Intrinsic_lora_sampling:
    def __init__(self):
        self.loaded_lora = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("intrinsic_loras"), ),
                "task": (
                [   
                    'depth map',
                    'surface normals',
                    'albedo',
                    'shading',
                ],
                {
                "default": 'depth map'
                    }),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
        },
            "optional": {
            "image": ("IMAGE",),
            "optional_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)
    FUNCTION = "onestepsample"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Sampler to use the intrinsic loras:  
https://github.com/duxiaodan/intrinsic-lora  
These LoRAs are tiny and thus included  
with this node pack.
"""

    def onestepsample(self, model, lora_name, clip, vae, text, task, per_batch, image=None, optional_latent=None):
        pbar = ProgressBar(3)

        if optional_latent is None:
            image_list = []
            for start_idx in range(0, image.shape[0], per_batch):
                sub_pixels = vae.vae_encode_crop_pixels(image[start_idx:start_idx+per_batch])
                image_list.append(vae.encode(sub_pixels[:,:,:,:3]))
            sample = torch.cat(image_list, dim=0)
        else:
            sample = optional_latent["samples"]
        noise = torch.zeros(sample.size(), dtype=sample.dtype, layout=sample.layout, device="cpu")
        prompt = task + "," + text
        positive, = CLIPTextEncode.encode(self, clip, prompt)
        negative = positive #negative shouldn't do anything in this scenario

        pbar.update(1)
     
        #custom model sampling to pass latent through as it is
        class X0_PassThrough(comfy.model_sampling.EPS):
            def calculate_denoised(self, sigma, model_output, model_input):
                return model_output
            def calculate_input(self, sigma, noise):
                return noise
        sampling_base = comfy.model_sampling.ModelSamplingDiscrete
        sampling_type = X0_PassThrough

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass
        model_sampling = ModelSamplingAdvanced(model.model.model_config)

        #load lora
        model_clone = model.clone()
        lora_path = folder_paths.get_full_path("intrinsic_loras", lora_name)        
        lora = load_torch_file(lora_path, safe_load=True)
        self.loaded_lora = (lora_path, lora)

        model_clone_with_lora = comfy.sd.load_lora_for_models(model_clone, None, lora, 1.0, 0)[0]

        model_clone_with_lora.add_object_patch("model_sampling", model_sampling)

        samples = {"samples": comfy.sample.sample(model_clone_with_lora, noise, 1, 1.0, "euler", "simple", positive, negative, sample,
                                  denoise=1.0, disable_noise=True, start_step=0, last_step=1,
                                  force_full_denoise=True, noise_mask=None, callback=None, disable_pbar=True, seed=None)}
        pbar.update(1)

        decoded = []
        for start_idx in range(0, samples["samples"].shape[0], per_batch):
            decoded.append(vae.decode(samples["samples"][start_idx:start_idx+per_batch]))
        image_out = torch.cat(decoded, dim=0)

        pbar.update(1)

        if task == 'depth map':
            imax = image_out.max()
            imin = image_out.min()
            image_out = (image_out-imin)/(imax-imin)
            image_out = torch.max(image_out, dim=3, keepdim=True)[0].repeat(1, 1, 1, 3)
        elif task == 'surface normals':
            image_out = F.normalize(image_out * 2 - 1, dim=3) / 2 + 0.5
            image_out = 1.0 - image_out
        else:
            image_out = image_out.clamp(-1.,1.)
            
        return (image_out, samples,)

```
