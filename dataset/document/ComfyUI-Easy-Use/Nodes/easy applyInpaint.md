---
tags:
- DepthMap
- Image
- Inpaint
---

# Easy Apply Inpaint
## Documentation
- Class name: `easy applyInpaint`
- Category: `EasyUse/Inpaint`
- Output node: `False`

This node facilitates the application of various inpainting techniques to images, offering a flexible approach to restoring or modifying specific areas of an image based on the inpainting mode selected. It integrates seamlessly with different inpainting strategies, such as 'fooocus_inpaint', 'brushnet_random', and others, to accommodate diverse inpainting needs.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration, serving as the foundational structure for the inpainting process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`image`**
    - The image to be inpainted, acting as the primary input for the inpainting operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`mask`**
    - A mask that specifies the areas of the image to be inpainted, guiding the inpainting algorithm on where to apply its effects.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`inpaint_mode`**
    - Determines the inpainting strategy to be used, offering options like 'normal', 'fooocus_inpaint', 'brushnet_random', and others to suit various inpainting scenarios.
    - Comfy dtype: `['normal', 'fooocus_inpaint', 'brushnet_random', 'brushnet_segmentation', 'powerpaint']`
    - Python dtype: `str`
- **`encode`**
    - Specifies the encoding method to prepare the image for inpainting, with options including 'none', 'vae_encode_inpaint', and others, affecting the preprocessing step.
    - Comfy dtype: `['none', 'vae_encode_inpaint', 'inpaint_model_conditioning', 'different_diffusion']`
    - Python dtype: `str`
- **`grow_mask_by`**
    - Defines how much to expand the mask beyond its original boundaries, which can influence the inpainting results by altering the area considered for inpainting.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`dtype`**
    - Specifies the data type for the computation, affecting performance and precision. Options include 'float16', 'bfloat16', 'float32', 'float64'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`fitting`**
    - Adjusts the fitting level of the inpainting to the surrounding image area, influencing how seamlessly the inpainted area blends with the rest of the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`function`**
    - Selects the specific inpainting function to use, such as 'text guided', 'shape guided', or 'object removal', tailoring the inpainting process to the content of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale`**
    - Controls the scale of the inpainting effect, which can affect the intensity and visibility of the inpainting within the specified area.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Determines the starting point of the inpainting process within the image, allowing for precise control over the area of effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at`**
    - Specifies the ending point of the inpainting process, defining the scope of the area to be inpainted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The modified pipeline configuration after applying the inpainting process, reflecting any changes made to the image.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class applyInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "inpaint_mode": (('normal', 'fooocus_inpaint', 'brushnet_random', 'brushnet_segmentation', 'powerpaint'),),
                "encode": (('none', 'vae_encode_inpaint', 'inpaint_model_conditioning', 'different_diffusion'), {"default": "none"}),
                "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'],),
                "fitting": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def inpaint_model_conditioning(self, pipe, image, vae, mask, grow_mask_by):
        if grow_mask_by >0:
            mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
        positive, negative, latent = InpaintModelConditioning().encode(pipe['positive'], pipe['negative'], image,
                                                                       vae, mask)
        pipe['positive'] = positive
        pipe['negative'] = negative
        pipe['samples'] = latent

        return pipe

    def get_brushnet_model(self, type, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if type == 'brushnet_random':
            brush_model = BRUSHNET_MODELS['random_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.random.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.random.mask.*\.(safetensors|bin)$'
        elif type == 'brushnet_segmentation':
            brush_model = BRUSHNET_MODELS['segmentation_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.segmentation.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.segmentation.mask.*\.(safetensors|bin)$'


        brushfile = [e for e in folder_paths.get_filename_list('inpaint') if re.search(pattern, e, re.IGNORECASE)]
        brushname = brushfile[0] if brushfile else None
        if not brushname:
            from urllib.parse import urlparse
            get_local_filepath(brush_model, INPAINT_DIR)
            parsed_url = urlparse(brush_model)
            brushname = os.path.basename(parsed_url.path)
        return brushname

    def get_powerpaint_model(self, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if model_type == 'sdxl':
            raise Exception("Powerpaint not supported for SDXL models")

        powerpaint_model = POWERPAINT_MODELS['v2.1']['model_url']
        powerpaint_clip = POWERPAINT_MODELS['v2.1']['clip_url']

        from urllib.parse import urlparse
        get_local_filepath(powerpaint_model, os.path.join(INPAINT_DIR, 'powerpaint'))
        model_parsed_url = urlparse(powerpaint_model)
        clip_parsed_url = urlparse(powerpaint_clip)
        model_name = os.path.join("powerpaint",os.path.basename(model_parsed_url.path))
        clip_name = os.path.join("powerpaint",os.path.basename(clip_parsed_url.path))
        return model_name, clip_name

    def apply(self, pipe, image, mask, inpaint_mode, encode, grow_mask_by, dtype, fitting, function, scale, start_at, end_at):
        new_pipe = {
            **pipe,
        }
        del pipe
        if inpaint_mode in ['brushnet_random', 'brushnet_segmentation']:
            brushnet = self.get_brushnet_model(inpaint_mode, new_pipe['model'])
            new_pipe, = applyBrushNet().apply(new_pipe, image, mask, brushnet, dtype, scale, start_at, end_at)
        elif inpaint_mode == 'powerpaint':
            powerpaint_model, powerpaint_clip = self.get_powerpaint_model(new_pipe['model'])
            new_pipe, = applyPowerPaint().apply(new_pipe, image, mask, powerpaint_model, powerpaint_clip, dtype, fitting, function, scale, start_at, end_at)

        vae = new_pipe['vae']
        if encode == 'none':
            if inpaint_mode == 'fooocus_inpaint':
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
        elif encode == 'vae_encode_inpaint':
            latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
            new_pipe['samples'] = latent
            if inpaint_mode == 'fooocus_inpaint':
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
        elif encode == 'inpaint_model_conditioning':
            if inpaint_mode == 'fooocus_inpaint':
                latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
                new_pipe['samples'] = latent
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, 0)
            else:
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, grow_mask_by)
        elif encode == 'different_diffusion':
            if inpaint_mode == 'fooocus_inpaint':
                latent, = VAEEncodeForInpaint().encode(vae, image, mask, grow_mask_by)
                new_pipe['samples'] = latent
                model, = applyFooocusInpaint().apply(new_pipe['model'], new_pipe['samples'],
                                                     list(FOOOCUS_INPAINT_HEAD.keys())[0],
                                                     list(FOOOCUS_INPAINT_PATCH.keys())[0])
                new_pipe['model'] = model
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, 0)
            else:
                new_pipe = self.inpaint_model_conditioning(new_pipe, image, vae, mask, grow_mask_by)
            cls = ALL_NODE_CLASS_MAPPINGS['DifferentialDiffusion']
            if cls is not None:
                model, = cls().apply(new_pipe['model'])
                new_pipe['model'] = model
            else:
                raise Exception("Differential Diffusion not found,please update comfyui")

        return (new_pipe,)

```
