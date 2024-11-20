---
tags:
- Sampling
---

# ∞ KSampler Image Generation Tool
## Documentation
- Class name: `LLMKSamplerTool`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

This node is designed to facilitate the generation of language models by providing a toolkit for sampling. It abstracts the complexities involved in sampling processes, making it easier to generate language models with specific characteristics or based on certain criteria.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for sampling, setting the foundation for the generation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`clip`**
    - Specifies the CLIP model to be used alongside the main model for enhanced conditioning.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`vae`**
    - Specifies the VAE model to be used in the generation process, contributing to the visual aspects of the generated model.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`width`**
    - Defines the width of the generated images, affecting the dimensions of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Defines the height of the generated images, affecting the dimensions of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - Determines the randomness seed for sampling, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to perform in the sampling process, affecting the detail and quality of the generated model.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, influencing the variance and creativity of the generated output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampling algorithm to use, tailoring the generation process to specific needs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduler for controlling the sampling process, further customizing the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Adjusts the level of denoising applied during the sampling process, affecting the clarity of the generated model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`optional_control_net`**
    - Optionally specifies a ControlNet model for additional control over the generation process.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `str`
- **`optional_control_image`**
    - Optionally includes a control image to guide the visual aspects of the generated model.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`optional_latent_image`**
    - Optionally incorporates a latent image to influence the visual aspects of the generated model.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`cn_strength`**
    - Specifies the strength of the ControlNet's influence on the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_start_percent`**
    - Defines the starting point of the ControlNet's influence in terms of the generation process's progress percentage.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cn_end_percent`**
    - Defines the ending point of the ControlNet's influence in terms of the generation process's progress percentage.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`tool`**
    - Comfy dtype: `TOOL`
    - Outputs the tool encapsulating the result of the sampling process, designed for generating language models.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMKSamplerTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (KSampler.SAMPLERS,),
                "scheduler": (KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }, 
            "optional": {
                "optional_control_net": ("CONTROL_NET",),
                "optional_control_image": ("IMAGE",),
                "optional_latent_image": ("LATENT",),
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "cn_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cn_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            }
        }

    RETURN_TYPES = ("TOOL",)
    FUNCTION = "image_generator_tool"

    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Tools"

    def image_generator_tool(self, model, clip, vae, width, height, seed, steps, cfg, 
                             sampler_name, scheduler, denoise, optional_control_net=None, 
                             optional_control_image=None, optional_latent_image=None, 
                             cn_strength=1.0, cn_start_percent=0.0, cn_end_percent=1.0):
        temp_path = get_full_path(0, "image_generator")
        os.makedirs(temp_path, exist_ok=True)

        def image_generator(pos_prompt: str, neg_prompt: str, gen_seed: int = 42, batch_size: int = 1) -> str:
            """
            Generates images based on provided prompts using a latent diffusion model.

            Parameters:
            pos_prompt (str): The positive prompt text that guides the image generation.
            neg_prompt (str): The negative prompt text that guides what should be avoided in the image generation.
            batch_size (int): The amount of images to generate.

            Returns:
            str: String response containing <img [path]> data.
            """
            try:
                gen_seed = int(gen_seed) if gen_seed else seed

                # Generate an empty latent imagen or use optional input
                if not optional_latent_image:
                    logger.info("Creating blank latent")
                    latent_image_generator = EmptyLatentImage()
                    latent_input = latent_image_generator.generate(width, height, batch_size=batch_size)[0]
                else:
                    logger.info("Using optional latent")
                    latent_input = optional_latent_image

                # Encode positive and negative texts with CLIP
                clip_encoder = CLIPTextEncode()
                positive_cond = clip_encoder.encode(clip, pos_prompt)[0]
                negative_cond = clip_encoder.encode(clip, neg_prompt)[0]

                # Apply optional Control Net
                if optional_control_net and isinstance(optional_control_image, torch.Tensor):
                    logger.info("Applying control net")
                    control_net = ControlNetApplyAdvanced()
                    cn_positive_cond, cn_negative_cond = control_net.apply_controlnet(
                        positive=positive_cond, 
                        negative=negative_cond, 
                        control_net=optional_control_net, 
                        image=optional_control_image, 
                        strength=cn_strength,
                        start_percent=cn_start_percent, 
                        end_percent=cn_end_percent
                    )
                    if cn_positive_cond and cn_negative_cond:
                        positive_cond = cn_positive_cond
                        negative_cond = cn_negative_cond

                # Sample from the latent image using common_ksampler
                logger.info("Generating...")
                latent_image = common_ksampler(
                    model, gen_seed, steps, cfg, sampler_name, scheduler, positive_cond, negative_cond, latent_input, denoise=denoise
                )[0]

                # Decode the samples
                logger.info("Decoding samples...")
                vae_decoder = VAEDecode()
                tensor_image = vae_decoder.decode(vae, latent_image)[0]

                # Save the images to temp location
                image_paths = []
                for image in tensor_image:
                    filename = f"{str(uuid.uuid4())}.png"
                    logger.info(f"Saving image {filename}")
                    image_path = os.path.join(temp_path, filename)
                    pil_image = tensor2pil(image)
                    pil_image.save(image_path)
                    image_paths.append(image_path)

            except Exception as e:
                return traceback.format_exc()
            
            # Create image path tags response
            out = ""
            for path in image_paths:
                out += f"<img {path}>\n"

            return out
        
        tool = {
                "name": "image_generator",
                "description": "A function that allows you to generate image(s). You can provide the `pos_prompt` which describes what should be in the image, and a `neg_prompt` which descibes what shouldn't be in the image. There is also an optional `batch_size` param which is how many images you want to generate.",
                "function": image_generator
        }

        return (tool, )

```
