# Easy Apply ICLight
## Documentation
- Class name: `easy icLightApply`
- Category: `EasyUse/Adapter`
- Output node: `True`

This node applies the ICLight model to enhance or modify the lighting of an image based on the specified mode. It leverages advanced image processing techniques to adjust the lighting conditions of the input image, making it suitable for various applications such as image editing, enhancement, and realistic rendering.
## Input types
### Required
- **`mode`**
    - Specifies the lighting mode to apply to the image, offering options like left light, right light, top light, etc., to simulate different lighting conditions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`model`**
    - The model to be used for the lighting adjustment process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`image`**
    - The input image to which the lighting effects will be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - A variational autoencoder used for encoding the image before applying the lighting effects.
    - Comfy dtype: `VAE`
    - Python dtype: `torch.nn.Module`
- **`lighting`**
    - Defines the specific lighting effect to be applied.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`source`**
    - Specifies the source mode for generating the lighting image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`remove_bg`**
    - Indicates whether the background of the image should be removed before applying the lighting effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model used after applying the lighting effects.
    - Python dtype: `str`
- **`lighting_image`**
    - Comfy dtype: `IMAGE`
    - The output image with the applied lighting effects.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class icLightApply:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (list(IC_LIGHT_MODELS.keys()),),
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "lighting": (['None', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light', 'Circle Light'],{"default": "None"}),
                "source": (['Use Background Image', 'Use Flipped Background Image', 'Left Light', 'Right Light', 'Top Light', 'Bottom Light', 'Ambient'],{"default": "Use Background Image"}),
                "remove_bg": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "lighting_image")
    FUNCTION = "apply"
    OUTPUT_NODE = True
    CATEGORY = "EasyUse/Adapter"

    def batch(self, image1, image2):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bilinear",
                                                "center").movedim(1, -1)
        s = torch.cat((image1, image2), dim=0)
        return s

    def removebg(self, image):
        if "easy imageRemBg" not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception("Please re-install ComfyUI-Easy-Use")
        cls = ALL_NODE_CLASS_MAPPINGS['easy imageRemBg']
        results = cls().remove('RMBG-1.4', image, 'Hide', 'ComfyUI')
        if "result" in results:
            image, _ = results['result']
            return image

    def apply(self, mode, model, image, vae, lighting, source, remove_bg):
        model_type = get_sd_version(model)
        if model_type == 'sdxl':
            raise Exception("IC Light model is not supported for SDXL now")

        batch_size, height, width, channel = image.shape
        if channel == 3:
            # remove bg
            if mode == 'Foreground' or batch_size == 1:
                if remove_bg:
                    image = self.removebg(image)
                else:
                    mask = torch.full((1, height, width), 1.0, dtype=torch.float32, device="cpu")
                    image, = JoinImageWithAlpha().join_image_with_alpha(image, mask)

        iclight = ICLight()
        if mode == 'Foreground':
          lighting_image = iclight.generate_lighting_image(image, lighting)
        else:
          lighting_image = iclight.generate_source_image(image, source)
          if source not in ['Use Background Image', 'Use Flipped Background Image']:
              _, height, width, _ = lighting_image.shape
              mask = torch.full((1, height, width), 1.0, dtype=torch.float32, device="cpu")
              lighting_image, = JoinImageWithAlpha().join_image_with_alpha(lighting_image, mask)
              if batch_size < 2:
                image = self.batch(image, lighting_image)
              else:
                original_image = [img.unsqueeze(0) for img in image]
                original_image = self.removebg(original_image[0])
                image = self.batch(original_image, lighting_image)

        latent, = VAEEncodeArgMax().encode(vae, image)
        key = 'iclight_' + mode + '_' + model_type
        model_path = get_local_filepath(IC_LIGHT_MODELS[mode]['sd1']["model_url"],
                                        os.path.join(folder_paths.models_dir, "unet"))
        ic_model = None
        if key in backend_cache.cache:
            log_node_info("easy icLightApply", f"Using icLightModel {mode+'_'+model_type} Cached")
            _, ic_model = backend_cache.cache[key][1]
            m, _ = iclight.apply(model_path, model, latent, ic_model)
        else:
            m, ic_model = iclight.apply(model_path, model, latent, ic_model)
            backend_cache.update_cache(key, 'iclight', (False, ic_model))
        return (m, lighting_image)

```
