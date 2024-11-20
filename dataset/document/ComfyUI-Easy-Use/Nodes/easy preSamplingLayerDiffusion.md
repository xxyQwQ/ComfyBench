---
tags:
- LayeredDiffusion
---

# PreSampling (LayerDiffuse)
## Documentation
- Class name: `easy preSamplingLayerDiffusion`
- Category: `EasyUse/PreSampling`
- Output node: `True`

This node is designed to apply a layer diffusion process to images or latent spaces before sampling in a generative model pipeline. It adjusts the blending and diffusion of foreground and background elements based on specified methods and weights, enhancing the control over the generation process. This node is part of the EasyUse/PreSampling category, focusing on preprocessing steps that modify the input data to achieve desired visual effects or characteristics in the final output.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration, including settings and states that affect the overall sampling and generation process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`method`**
    - Specifies the method used for layer diffusion, determining how foreground and background elements are blended or diffused.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `LayerMethod`
- **`weight`**
    - Controls the intensity of the diffusion effect, allowing for fine-tuning of the blend between elements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Defines the number of steps to execute in the diffusion process, affecting the depth of the effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings that influence the sampling behavior, such as noise levels and model parameters.
    - Comfy dtype: `FLOAT`
    - Python dtype: `dict`
- **`sampler_name`**
    - The name of the sampler to be used, influencing the pattern and quality of the generated output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduling algorithm for the sampling process, affecting the progression of the generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Adjusts the level of denoising applied during the diffusion process, impacting the clarity and detail of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image`**
    - An optional image input for the diffusion process, allowing for direct manipulation of existing images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`blended_image`**
    - An optional input for providing a pre-blended image, used in conjunction with specific diffusion methods.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`mask`**
    - An optional mask input, used to define areas of the image that should be treated differently during the diffusion process.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The modified pipeline configuration after applying the layer diffusion process, ready for further processing or generation steps.
    - Python dtype: `dict`
- **`ui`**
    - Provides a user interface component, typically displaying the seed value used in the generation process.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class layerDiffusionSettings:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
            {
             "pipe": ("PIPE_LINE",),
             "method": ([LayerMethod.FG_ONLY_ATTN.value, LayerMethod.FG_ONLY_CONV.value, LayerMethod.EVERYTHING.value, LayerMethod.FG_TO_BLEND.value, LayerMethod.BG_TO_BLEND.value],),
             "weight": ("FLOAT",{"default": 1.0, "min": -1, "max": 3, "step": 0.05},),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS+ new_schedulers, {"default": "normal"}),
             "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
             "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
             },
            "optional": {
                "image": ("IMAGE",),
                "blended_image": ("IMAGE",),
                "mask": ("MASK",),
                # "latent": ("LATENT",),
                # "blended_latent": ("LATENT",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def get_layer_diffusion_method(self, method, has_blend_latent):
        method = LayerMethod(method)
        if has_blend_latent:
            if method == LayerMethod.BG_TO_BLEND:
                method = LayerMethod.BG_BLEND_TO_FG
            elif method == LayerMethod.FG_TO_BLEND:
                method = LayerMethod.FG_BLEND_TO_BG
        return method

    def settings(self, pipe, method, weight, steps, cfg, sampler_name, scheduler, denoise, seed, image=None, blended_image=None, mask=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        blend_samples = pipe['blend_samples'] if "blend_samples" in pipe else None
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1

        method = self.get_layer_diffusion_method(method, blend_samples is not None or blended_image is not None)

        if image is not None or "image" in pipe:
            image = image if image is not None else pipe['image']
            if mask is not None:
                print('inpaint')
                samples, = VAEEncodeForInpaint().encode(vae, image, mask)
            else:
                samples = {"samples": vae.encode(image[:,:,:,:3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image
        elif "samp_images" in pipe:
            samples = {"samples": vae.encode(pipe["samp_images"][:,:,:,:3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = pipe["samp_images"]
        else:
            if method not in [LayerMethod.FG_ONLY_ATTN, LayerMethod.FG_ONLY_CONV, LayerMethod.EVERYTHING]:
                raise Exception("image is missing")

            samples = pipe["samples"]
            images = pipe["images"]

        if method in [LayerMethod.BG_BLEND_TO_FG, LayerMethod.FG_BLEND_TO_BG]:
            if blended_image is None and blend_samples is None:
                raise Exception("blended_image is missing")
            elif blended_image is not None:
                blend_samples = {"samples": vae.encode(blended_image[:,:,:,:3])}
                blend_samples = RepeatLatentBatch().repeat(blend_samples, batch_size)[0]

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "blend_samples": blend_samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "enabled",
                "layer_diffusion_method": method,
                "layer_diffusion_weight": weight,
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

```
