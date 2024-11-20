---
tags:
- DepthMap
- Image
- Inpaint
---

# EasyKSampler (Inpainting)
## Documentation
- Class name: `easy kSamplerInpainting`
- Category: `EasyUse/Sampler`
- Output node: `True`

This node specializes in generating images by inpainting missing parts or modifying existing images. It leverages a variety of inpainting models and techniques to fill in gaps or replace sections of an image based on the surrounding context, allowing for creative and customized image manipulation.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration for the inpainting process, including the model and parameters to be used.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`grow_mask_by`**
    - Specifies how much to expand the inpainting mask by, in pixels, to ensure smoother transitions and better inpainting results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`image_output`**
    - Determines how the output image is handled, offering options for hiding, previewing, saving, or sending the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`link_id`**
    - An identifier for linking this operation with other processes or outputs, facilitating complex workflows.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_prefix`**
    - A prefix for naming saved images, allowing for organized output management.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`additional`**
    - Selects the specific inpainting model or technique to be used, providing flexibility in approach and customization of the inpainting effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`model`**
    - The inpainting model used for generating the output, integral to the inpainting process.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`mask`**
    - The mask that defines the area to be inpainted, crucial for guiding the inpainting process.
    - Comfy dtype: `MASK`
    - Python dtype: `object`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The updated pipeline configuration after the inpainting process, including any modifications to the model or parameters.
    - Python dtype: `dict`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after inpainting, showcasing the modifications or filled gaps.
    - Python dtype: `object`
- **`vae`**
    - Comfy dtype: `VAE`
    - The variational autoencoder used in the inpainting process, if applicable, indicating its role in generating or refining the output.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class samplerSimpleInpainting(samplerFull):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "grow_mask_by": ("INT", {"default": 6, "min": 0, "max": 64, "step": 1}),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 "additional": (["None", "InpaintModelCond", "Differential Diffusion", "Fooocus Inpaint", "Fooocus Inpaint + DD", "Brushnet Random", "Brushnet Random + DD", "Brushnet Segmentation", "Brushnet Segmentation + DD"],{"default": "None"})
                 },
                "optional": {
                    "model": ("MODEL",),
                    "mask": ("MASK",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }

    RETURN_TYPES = ("PIPE_LINE", "IMAGE", "VAE")
    RETURN_NAMES = ("pipe", "image", "vae")
    OUTPUT_NODE = True
    FUNCTION = "inpainting"
    CATEGORY = "EasyUse/Sampler"

    def dd(self, model, positive, negative, pixels, vae, mask):
        positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
        cls = ALL_NODE_CLASS_MAPPINGS['DifferentialDiffusion']
        if cls is not None:
            model, = cls().apply(model)
        else:
            raise Exception("Differential Diffusion not found,please update comfyui")
        return positive, negative, latent, model

    def get_brushnet_model(self, type, model):
        model_type = 'sdxl' if isinstance(model.model.model_config, comfy.supported_models.SDXL) else 'sd1'
        if type == 'random':
            brush_model = BRUSHNET_MODELS['random_mask'][model_type]['model_url']
            if model_type == 'sdxl':
                pattern = 'brushnet.random.mask.sdxl.*\.(safetensors|bin)$'
            else:
                pattern = 'brushnet.random.mask.*\.(safetensors|bin)$'
        elif type == 'segmentation':
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

    def apply_brushnet(self, brushname, model, vae, image, mask, positive, negative, scale=1.0, start_at=0, end_at=10000):
        if "BrushNetLoader" not in ALL_NODE_CLASS_MAPPINGS:
            raise Exception("BrushNetLoader not found,please install ComfyUI-BrushNet")
        cls = ALL_NODE_CLASS_MAPPINGS['BrushNetLoader']
        brushnet, = cls().brushnet_loading(brushname, 'float16')
        cls = ALL_NODE_CLASS_MAPPINGS['BrushNet']
        m, positive, negative, latent = cls().model_update(model=model, vae=vae, image=image, mask=mask, brushnet=brushnet, positive=positive, negative=negative, scale=scale, start_at=start_at, end_at=end_at)
        return m, positive, negative, latent

    def inpainting(self, pipe, grow_mask_by, image_output, link_id, save_prefix, additional, model=None, mask=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        _model = model if model is not None else pipe['model']
        latent = pipe['samples'] if 'samples' in pipe else None
        positive = pipe['positive']
        negative = pipe['negative']
        images = pipe["images"] if pipe and "images" in pipe else None
        vae = pipe["vae"] if pipe and "vae" in pipe else None
        if 'noise_mask' in latent and mask is None:
            mask = latent['noise_mask']
        elif mask is not None:
            if images is None:
                raise Exception("No Images found")
            if vae is None:
                raise Exception("No VAE found")

        match additional:
            case 'Differential Diffusion':
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case 'InpaintModelCond':
                if mask is not None:
                    mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                positive, negative, latent = InpaintModelConditioning().encode(positive, negative, images, vae, mask)
            case 'Fooocus Inpaint':
                head = list(FOOOCUS_INPAINT_HEAD.keys())[0]
                patch = list(FOOOCUS_INPAINT_PATCH.keys())[0]
                if mask is not None:
                    latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)
                _model, = applyFooocusInpaint().apply(_model, latent, head, patch)
            case 'Fooocus Inpaint + DD':
                head = list(FOOOCUS_INPAINT_HEAD.keys())[0]
                patch = list(FOOOCUS_INPAINT_PATCH.keys())[0]
                if mask is not None:
                    latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)
                _model, = applyFooocusInpaint().apply(_model, latent, head, patch)
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case 'Brushnet Random':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('random', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
            case 'Brushnet Random + DD':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('random', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case 'Brushnet Segmentation':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('segmentation', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
            case 'Brushnet Segmentation + DD':
                mask, = GrowMask().expand_mask(mask, grow_mask_by, False)
                brush_name = self.get_brushnet_model('segmentation', _model)
                _model, positive, negative, latent = self.apply_brushnet(brush_name, _model, vae, images, mask, positive, negative)
                positive, negative, latent, _model = self.dd(_model, positive, negative, images, vae, mask)
            case _:
                latent, = VAEEncodeForInpaint().encode(vae, images, mask, grow_mask_by)

        results = super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, _model, positive, negative, latent, vae, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

        result = results['result']

        return {"ui":results['ui'],"result":(result[0], result[1], result[0]['vae'],)}

```
