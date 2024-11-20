---
tags:
- Sampling
---

# KSampler Adv. (Efficient)
## Documentation
- Class name: `KSampler Adv. (Efficient)`
- Category: `Efficiency Nodes/Sampling`
- Output node: `True`

This node specializes in advanced sampling techniques for generating or refining images based on given inputs. It leverages a more sophisticated approach compared to regular samplers, allowing for enhanced control over the sampling process, including noise addition and stepwise refinement, to produce high-quality outputs.
## Input types
### Required
- **`model`**
    - The model parameter specifies the generative model to be used for sampling. It is crucial for determining the behavior and quality of the generated images.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`add_noise`**
    - Determines whether noise should be added to the sampling process, allowing for more variability and realism in the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`noise_seed`**
    - Seed for the noise generation, ensuring control over the randomness in the added noise for reproducibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Specifies the number of steps to perform in the sampling process, affecting the detail and quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The CFG scale used to balance between fidelity to the prompt and the randomness of the generation, influencing the output's creativity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler to use, defining the specific sampling algorithm and its characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler for controlling the sampling process, affecting how the steps are executed over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - Positive conditioning to guide the generation towards desired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Negative conditioning to steer the generation away from certain attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent_image`**
    - The initial latent image to start the sampling process from, setting the basis for generation or refinement.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`start_at_step`**
    - The step number to start the sampling process, allowing for mid-process interventions or adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - The final step number in the sampling process, defining the endpoint for generation or refinement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`return_with_leftover_noise`**
    - Indicates whether to return the sampled image with leftover noise, offering a choice between a cleaner or more raw output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`preview_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`vae_decode`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
### Optional
- **`optional_vae`**
    - unknown
    - Comfy dtype: `VAE`
    - Python dtype: `unknown`
- **`script`**
    - unknown
    - Comfy dtype: `SCRIPT`
    - Python dtype: `unknown`
## Output types
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - The output includes the model used in the sampling process, potentially modified or updated.
    - Python dtype: `torch.nn.Module`
- **`CONDITIONING+`**
    - Comfy dtype: `CONDITIONING`
    - Output conditioning that positively influences the generation towards desired characteristics.
    - Python dtype: `str`
- **`CONDITIONING-`**
    - Comfy dtype: `CONDITIONING`
    - Output conditioning that negatively influences the generation, steering it away from undesired characteristics.
    - Python dtype: `str`
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The latent representation of the image after the sampling process, ready for further processing or conversion to image form.
    - Python dtype: `torch.Tensor`
- **`VAE`**
    - Comfy dtype: `VAE`
    - The VAE model used or affected during the sampling process, if applicable.
    - Python dtype: `torch.nn.Module`
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The final image output after the sampling process, reflecting the combined effects of conditioning, noise, and model parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [KSampler Adv. (Efficient)](../../efficiency-nodes-comfyui/Nodes/KSampler Adv. (Efficient).md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [ReActorFaceSwap](../../comfyui-reactor-node/Nodes/ReActorFaceSwap.md)
    - [VAEEncodeTiled](../../Comfy/Nodes/VAEEncodeTiled.md)
    - [SaveImage](../../Comfy/Nodes/SaveImage.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)



## Source code
```python
class TSC_KSamplerAdvanced(TSC_KSampler):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"model": ("MODEL",),
                     "add_noise": (["enable", "disable"],),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": (["disable", "enable"],),
                     "preview_method": (["auto", "latent2rgb", "taesd", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false", "output only", "output only (tiled)"],),
                     },
                "optional": {"optional_vae": ("VAE",),
                             "script": ("SCRIPT",), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "sample_adv"
    CATEGORY = "Efficiency Nodes/Sampling"

    def sample_adv(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, start_at_step, end_at_step, return_with_leftover_noise, preview_method, vae_decode,
               prompt=None, extra_pnginfo=None, my_unique_id=None, optional_vae=(None,), script=None):

        return super().sample(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, preview_method, vae_decode, denoise=1.0, prompt=prompt, extra_pnginfo=extra_pnginfo, my_unique_id=my_unique_id,
               optional_vae=optional_vae, script=script, add_noise=add_noise, start_at_step=start_at_step,end_at_step=end_at_step,
                       return_with_leftover_noise=return_with_leftover_noise,sampler_type="advanced")

```
