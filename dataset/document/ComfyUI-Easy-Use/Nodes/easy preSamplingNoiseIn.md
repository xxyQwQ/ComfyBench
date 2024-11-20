---
tags:
- LatentNoise
- Noise
---

# PreSampling (NoiseIn)
## Documentation
- Class name: `easy preSamplingNoiseIn`
- Category: `EasyUse/PreSampling`
- Output node: `True`

This node is designed to inject noise into latent representations before the sampling process in a customizable manner. It allows for the adjustment of noise characteristics and the application of noise based on specific conditions or parameters, enhancing the flexibility and control over the noise injection process in generative models.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration and state, serving as the context for noise injection and other processing steps.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`factor`**
    - Adjusts the intensity of the noise injected into the latent representation, allowing for finer control over the noise level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Determines the number of steps for the sampling process, impacting the intensity and granularity of noise injection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the configuration strength, influencing the balance between the original content and the generated aspects in the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the sampler to be used, allowing for different sampling strategies and their corresponding noise characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Defines the scheduling strategy for noise injection, enabling precise control over the timing and sequence of noise application.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `object`
- **`denoise`**
    - Adjusts the denoising factor, affecting the level of noise reduction applied to the latent representation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seed`**
    - Sets the seed for random number generation, ensuring consistency and reproducibility in noise injection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_noise_seed`**
    - An optional seed for generating noise, providing an additional layer of control over the randomness of the noise injected.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`optional_latent`**
    - An optional latent representation that can be directly manipulated or injected with noise, offering more flexibility in the noise injection process.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The modified latent representation with noise injected, ready for further processing or sampling in the pipeline.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class samplerSettingsNoiseIn:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS+new_schedulers,),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional": {
                    "optional_noise_seed": ("INT",{"forceInput": True}),
                    "optional_latent": ("LATENT",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def slerp(self, val, low, high):
        dims = low.shape

        low = low.reshape(dims[0], -1)
        high = high.reshape(dims[0], -1)

        low_norm = low / torch.norm(low, dim=1, keepdim=True)
        high_norm = high / torch.norm(high, dim=1, keepdim=True)

        low_norm[low_norm != low_norm] = 0.0
        high_norm[high_norm != high_norm] = 0.0

        omega = torch.acos((low_norm * high_norm).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(
            1) * high

        return res.reshape(dims)

    def prepare_mask(self, mask, shape):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                                               size=(shape[2], shape[3]), mode="bilinear")
        mask = mask.expand((-1, shape[1], -1, -1))
        if mask.shape[0] < shape[0]:
            mask = mask.repeat((shape[0] - 1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        return mask

    def expand_mask(self, mask, expand, tapered_corners):
        try:
            import scipy

            c = 0 if tapered_corners else 1
            kernel = np.array([[c, 1, c],
                               [1, 1, 1],
                               [c, 1, c]])
            mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
            out = []
            for m in mask:
                output = m.numpy()
                for _ in range(abs(expand)):
                    if expand < 0:
                        output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                    else:
                        output = scipy.ndimage.grey_dilation(output, footprint=kernel)
                output = torch.from_numpy(output)
                out.append(output)

            return torch.stack(out, dim=0)
        except:
            return None

    def settings(self, pipe, factor, steps, cfg, sampler_name, scheduler, denoise, seed, optional_noise_seed=None, optional_latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        latent = optional_latent if optional_latent is not None else pipe["samples"]
        model = pipe["model"]

        # generate base noise
        batch_size, _, height, width = latent["samples"].shape
        generator = torch.manual_seed(seed)
        base_noise = torch.randn((1, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).repeat(batch_size, 1, 1, 1).cpu()

        # generate variation noise
        if optional_noise_seed is None or optional_noise_seed == seed:
            optional_noise_seed = seed+1
        generator = torch.manual_seed(optional_noise_seed)
        variation_noise = torch.randn((batch_size, 4, height, width), dtype=torch.float32, device="cpu",
                                      generator=generator).cpu()

        slerp_noise = self.slerp(factor, base_noise, variation_noise)

        end_at_step = steps  # min(steps, end_at_step)
        start_at_step = round(end_at_step - end_at_step * denoise)

        device = comfy.model_management.get_torch_device()
        comfy.model_management.load_model_gpu(model)
        model_patcher = comfy.model_patcher.ModelPatcher(model.model, load_device=device, offload_device=comfy.model_management.unet_offload_device())
        sampler = comfy.samplers.KSampler(model_patcher, steps=steps, device=device, sampler=sampler_name,
                                          scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.cpu().numpy()

        work_latent = latent.copy()
        work_latent["samples"] = latent["samples"].clone() + slerp_noise * sigma

        if "noise_mask" in latent:
            noise_mask = self.prepare_mask(latent["noise_mask"], latent['samples'].shape)
            work_latent["samples"] = noise_mask * work_latent["samples"] + (1-noise_mask) * latent["samples"]
            work_latent['noise_mask'] = self.expand_mask(latent["noise_mask"].clone(), 5, True)

        if pipe is None:
            pipe = {}

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": work_latent,
            "images": pipe['images'],
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "add_noise": "disable"
            }
        }

        return (new_pipe,)

```
