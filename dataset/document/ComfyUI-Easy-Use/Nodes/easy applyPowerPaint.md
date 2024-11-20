---
tags:
- DepthMap
- Image
---

# Easy Apply PowerPaint
## Documentation
- Class name: `easy applyPowerPaint`
- Category: `EasyUse/Inpaint`
- Output node: `False`

The `easy applyPowerPaint` node is designed to facilitate advanced image manipulation tasks such as text-guided editing, shape-guided creation, object removal, context-aware adjustments, and image outpainting. It leverages a combination of models and techniques, including PowerPaint, to apply complex transformations and enhancements to images based on user-defined parameters.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline state, including models and configurations, that the node operates on. It's essential for maintaining the context and progress of image manipulation tasks.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict[str, Any]`
- **`image`**
    - The input image to be manipulated or enhanced. It serves as the primary subject for the node's operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Any`
- **`mask`**
    - A mask that specifies the areas of the image to be edited or preserved. It plays a crucial role in targeted manipulations such as object removal or area-specific enhancements.
    - Comfy dtype: `MASK`
    - Python dtype: `Any`
- **`powerpaint_model`**
    - Specifies the PowerPaint model to be used for the image manipulation task. It determines the underlying algorithm and capabilities available for the operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`powerpaint_clip`**
    - Identifies the specific CLIP model associated with PowerPaint, used for understanding and interpreting the content and context of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`dtype`**
    - Defines the data type for the PowerPaint model's computations, affecting performance and precision.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`fitting`**
    - Controls the fitting level of the manipulation to the original image, allowing for fine-tuning the intensity and integration of the applied effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`function`**
    - Determines the specific manipulation function to be applied, such as text-guided editing or object removal, guiding the node's operation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale`**
    - Adjusts the scale of the manipulation effect, influencing the extent of changes applied to the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Specifies the starting point for the manipulation process, enabling phased or progressive image editing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at`**
    - Defines the endpoint for the manipulation process, allowing for controlled and precise editing operations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_memory`**
    - Optimizes memory usage during the manipulation process, catering to different system capabilities and requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The updated pipeline state after applying the PowerPaint manipulations, reflecting the changes and enhancements made to the image.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class applyPowerPaint:
    def get_files_with_extension(folder='inpaint', extensions='.safetensors'):
        return [file for file in folder_paths.get_filename_list(folder) if file.endswith(extensions)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "powerpaint_model": (s.get_files_with_extension(),),
                "powerpaint_clip": (s.get_files_with_extension(extensions='.bin'),),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'],),
                "fitting": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 1.0}),
                "function": (['text guided', 'shape guided', 'object removal', 'context aware', 'image outpainting'],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "save_memory": (['none', 'auto', 'max'],),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, pipe, image, mask, powerpaint_model, powerpaint_clip, dtype, fitting, function, scale, start_at, end_at, save_memory='none'):
        model = pipe['model']
        vae = pipe['vae']
        positive = pipe['positive']
        negative = pipe['negative']

        cls = BrushNet()
        # load powerpaint clip
        if powerpaint_clip in backend_cache.cache:
            log_node_info("easy powerpaintApply", f"Using {powerpaint_clip} Cached")
            _, ppclip = backend_cache.cache[powerpaint_clip][1]
        else:
            model_url = POWERPAINT_MODELS['base_fp16']['model_url']
            base_clip = get_local_filepath(model_url, os.path.join(folder_paths.models_dir, 'clip'))
            ppclip, = cls.load_powerpaint_clip(base_clip, os.path.join(folder_paths.get_full_path("inpaint", powerpaint_clip)))
            backend_cache.update_cache(powerpaint_clip, 'ppclip', (False, ppclip))
        # load powerpaint model
        if powerpaint_model in backend_cache.cache:
            log_node_info("easy powerpaintApply", f"Using {powerpaint_model} Cached")
            _, powerpaint = backend_cache.cache[powerpaint_model][1]
        else:
            powerpaint_file = os.path.join(folder_paths.get_full_path("inpaint", powerpaint_model))
            powerpaint, = cls.load_brushnet_model(powerpaint_file, dtype)
            backend_cache.update_cache(powerpaint_model, 'powerpaint', (False, powerpaint))
        m, positive, negative, latent = cls.powerpaint_model_update(model=model, vae=vae, image=image, mask=mask, powerpaint=powerpaint,
                                                           clip=ppclip, positive=positive,
                                                           negative=negative, fitting=fitting, function=function,
                                                           scale=scale, start_at=start_at, end_at=end_at, save_memory=save_memory)
        new_pipe = {
            **pipe,
            "model": m,
            "positive": positive,
            "negative": negative,
            "samples": latent,
        }
        del pipe
        return (new_pipe,)

```
