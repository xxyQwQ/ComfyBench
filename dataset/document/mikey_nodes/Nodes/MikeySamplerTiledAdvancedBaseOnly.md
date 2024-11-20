---
tags:
- Sampling
---

# Mikey Sampler Tiled Advanced Base Only
## Documentation
- Class name: `MikeySamplerTiledAdvancedBaseOnly`
- Category: `Mikey/Sampling`
- Output node: `False`

This node is designed for advanced tiled sampling without the need for smooth steps or a refiner. It focuses on generating or processing images in a tiled manner, optimizing for scenarios where seamless integration of tiles is crucial without additional refinement steps.
## Input types
### Required
- **`base_model`**
    - Specifies the base model used for sampling, setting the foundation for the generation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`samples`**
    - The initial latent samples to be processed or refined through the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The variational autoencoder used alongside the base model to process or refine samples.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`positive_cond_base`**
    - Positive conditioning for the base model to guide the sampling towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative_cond_base`**
    - Negative conditioning for the base model to steer the sampling away from undesired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`model_name`**
    - The name of the model, typically used to identify different models within a framework or library.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`seed`**
    - Seed for random number generation, ensuring reproducibility across sampling runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise_image`**
    - Specifies the degree of denoising applied to the image, affecting the clarity and quality of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - The number of steps to run in the sampling process, affecting the detail and quality of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, influencing the generation's adherence to the given conditions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler used, affecting the diversity and quality of generated samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler used to manage the sampling process, impacting the progression and variation of samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_by`**
    - The factor by which the image is upscaled, affecting the resolution and detail of the final output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tiler_denoise`**
    - Specifies the degree of denoising applied to each tile, affecting the consistency and quality of the tiled output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`tile_size`**
    - Defines the size of each tile in the generated image, affecting the granularity of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image_optional`**
    - An optional image input that can be used to influence the sampling process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`output_image`**
    - Comfy dtype: `IMAGE`
    - The generated image after the tiled sampling process, reflecting the combined influence of all input parameters.
    - Python dtype: `PIL.Image`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MikeySamplerTiledAdvancedBaseOnly:
    # there is no step skipped, so no smooth steps are required
    # also no refiner for this
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_model": ("MODEL",),
                             "samples": ("LATENT",), "vae": ("VAE",),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "denoise_image": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                             "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                             "cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                             "upscale_by": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                             "tiler_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})},
                "optional": {"image_optional": ("IMAGE",),}}

    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = 'Mikey/Sampling'

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                    upscale_by, model_name, seed, vae, denoise_image,
                    steps, cfg, sampler_name, scheduler):
            image_scaler = ImageScale()
            vaedecoder = VAEDecode()
            uml = UpscaleModelLoader()
            upscale_model = uml.load_model(model_name)[0]
            iuwm = ImageUpscaleWithModel()
            # step 1 run base model
            start_step = int(steps - (steps * denoise_image))
            #print(f'base model start_step: {start_step}, last_step: {last_step}')
            sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                    start_step=start_step, last_step=steps, force_full_denoise=False)[0]
            # step 3 upscale image using a simple AI image upscaler
            pixels = vaedecoder.decode(vae, sample1)[0]
            org_width, org_height = pixels.shape[2], pixels.shape[1]
            img = iuwm.upscale(upscale_model, image=pixels)[0]
            upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
            img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
            return img, upscaled_width, upscaled_height

    def upscale_image(self, samples, vae,
                    upscale_by, model_name):
            image_scaler = ImageScale()
            vaedecoder = VAEDecode()
            uml = UpscaleModelLoader()
            upscale_model = uml.load_model(model_name)[0]
            iuwm = ImageUpscaleWithModel()
            # step 3 upscale image using a simple AI image upscaler
            pixels = vaedecoder.decode(vae, samples)[0]
            org_width, org_height = pixels.shape[2], pixels.shape[1]
            img = iuwm.upscale(upscale_model, image=pixels)[0]
            upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
            img = image_scaler.upscale(img, 'nearest-exact', upscaled_width, upscaled_height, 'center')[0]
            return img, upscaled_width, upscaled_height

    def run(self, seed, base_model, vae, samples, positive_cond_base, negative_cond_base,
            model_name, upscale_by=2.0, tiler_denoise=0.4, tile_size=1024,
            upscale_method='normal', denoise_image=1.0, steps=30, cfg=6.5,
            sampler_name='dpmpp_sde_gpu', scheduler='karras', image_optional=None):
        # if image not none replace samples with decoded image
        if image_optional is not None:
            vaeencoder = VAEEncode()
            samples = vaeencoder.encode(vae, image_optional)[0]
        if denoise_image > 0:
            # phase 1: run base, refiner, then upscaler model
            img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                                                upscale_by, model_name, seed, vae, denoise_image,
                                                                steps, cfg, sampler_name, scheduler)
            img = tensor2pil(img)
        else:
            img = self.upscale_image(samples, vae, upscale_by, model_name)
            img = tensor2pil(img)
        # phase 2: run tiler
        tiled_image = run_tiler_for_steps(img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps, tiler_denoise, tile_size)
        return (tiled_image, )

```
