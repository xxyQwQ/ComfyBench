---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# hiresfixScale
## Documentation
- Class name: `ttN hiresfixScale`
- Category: `🌏 tinyterra/image`
- Output node: `True`

The ttN hiresfixScale node specializes in enhancing image resolution through various upscale methods, including model-based and algorithmic approaches. It provides options for rescaling images after model application, adjusting the scale by percentage or to specific dimensions while maintaining aspect ratios, and optionally outputting the result in latent space. This node is designed to improve image quality and detail, catering to both standard and custom upscale needs.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the upscale model to be used for enhancing the image resolution. It's essential for selecting the appropriate model-based upscale method.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae`**
    - The variational autoencoder used for decoding or encoding images when required, playing a crucial role in the upscale process.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`image`**
    - The input image to be upscaled. This parameter is the starting point of the upscale process and determines the initial quality and resolution.
    - Comfy dtype: `IMAGE`
    - Python dtype: `numpy.ndarray`
- **`rescale_after_model`**
    - Indicates whether the image should be rescaled after being processed by the upscale model, affecting the final image size and resolution.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`rescale_method`**
    - Defines the method used for rescaling the image after model upscaling, influencing the final appearance and dimensions of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`rescale`**
    - Specifies how the image should be rescaled, offering options like by percentage, to specific width/height, or to maintain aspect ratio by adjusting the longer side.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`percent`**
    - The percentage by which the image should be rescaled, relevant when the 'rescale' parameter is set to 'by percentage'. It directly impacts the final image size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - The target width for the image after rescaling, applicable when the 'rescale' parameter is set to 'to Width/Height'. It determines the final width of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The target height for the image after rescaling, applicable when the 'rescale' parameter is set to 'to Width/Height'. It determines the final height of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`longer_side`**
    - The target size for the longer side of the image when maintaining aspect ratio, affecting the final dimensions while preserving the image's original proportions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop`**
    - Determines if and how the upscaled image should be cropped, offering options like disabled or center cropping. This affects the final composition of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`image_output`**
    - Controls how the upscaled image is outputted, with options for hiding, previewing, saving, or a combination of hiding and saving. This affects how users interact with and access the final image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - The prefix added to the filename when saving the upscaled image, allowing for easy identification and organization of output files.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`output_latent`**
    - Indicates whether the result should also include the upscaled image in latent space, enabling further manipulation or analysis in this format.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The upscaled image in latent space, available when output_latent is set to True, allowing for further manipulation or analysis.
    - Python dtype: `Dict[str, Any]`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final upscaled image, showcasing the improved resolution and detail achieved through the upscale process.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_modelScale:
    version = '1.1.0'
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"),),
                              "vae": ("VAE",),
                              "image": ("IMAGE",),
                              "rescale_after_model": ([False, True],{"default": True}),
                              "rescale_method": (s.upscale_methods,),
                              "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect'],),
                              "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                              "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "crop": (s.crop_methods,),
                              "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                              "save_prefix": ("STRING", {"default": "ComfyUI"}),
                              "output_latent": ([False, True],{"default": True}),},
                "hidden": {   "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                               "ttNnodeVersion": ttN_modelScale.version},
        }
        
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", 'image',)

    FUNCTION = "upscale"
    CATEGORY = "🌏 tinyterra/image"
    OUTPUT_NODE = True

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def upscale(self, model_name, vae, image, rescale_after_model, rescale_method, rescale, percent, width, height, longer_side, crop, image_output, save_prefix, output_latent, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # Load Model
        upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(model_name)[0]

        # Model upscale
        s = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)[0]

        # Post Model Rescale
        if rescale_after_model == True:
            samples = s.movedim(-1, 1)
            orig_height = samples.shape[2]
            orig_width = samples.shape[3]
            if rescale == "by percentage" and percent != 0:
                height = percent / 100 * orig_height
                width = percent / 100 * orig_width
                if (width > MAX_RESOLUTION):
                    width = MAX_RESOLUTION
                if (height > MAX_RESOLUTION):
                    height = MAX_RESOLUTION

                width = ttNsampler.enforce_mul_of_64(width)
                height = ttNsampler.enforce_mul_of_64(height)
            elif rescale == "to longer side - maintain aspect":
                longer_side = ttNsampler.enforce_mul_of_64(longer_side)
                if orig_width > orig_height:
                    width, height = longer_side, ttNsampler.enforce_mul_of_64(longer_side * orig_height / orig_width)
                else:
                    width, height = ttNsampler.enforce_mul_of_64(longer_side * orig_width / orig_height), longer_side
                    

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1,-1)

        # vae encode
        if output_latent == True:
            pixels = self.vae_encode_crop_pixels(s)
            t = vae.encode(pixels[:,:,:,:3])
            if image_output == "return latent":
                return ({"samples":t})
        else:
            t = None

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)
        results = ttN_save.images(s, save_prefix, image_output)
        
        if image_output in ("Hide", "Hide/Save"):
            return ({"samples":t}, s,)

        return {"ui": {"images": results}, 
                "result": ({"samples":t}, s,)}

```
