# Easy Apply BrushNet
## Documentation
- Class name: `easy applyBrushNet`
- Category: `EasyUse/Inpaint`
- Output node: `False`

The `easy applyBrushNet` node integrates the functionality of BrushNet, a specialized neural network for image inpainting and enhancement, into a pipeline. It leverages BrushNet to apply sophisticated image processing techniques, enhancing or modifying images based on given masks and inputs. This node is designed to seamlessly incorporate BrushNet's capabilities into broader workflows, facilitating advanced image manipulation tasks.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline context, including models and configurations, that BrushNet will operate within. It's essential for providing the necessary environment and parameters for BrushNet's execution.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict[str, Any]`
- **`image`**
    - The target image to be processed or enhanced by BrushNet. It serves as the primary input for the image manipulation tasks.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A binary mask indicating the areas of the image to be inpainted or modified. It guides BrushNet in focusing its processing on specific regions of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`brushnet`**
    - Specifies the BrushNet model to be used for the image processing task, playing a crucial role in determining the specific techniques and enhancements applied to the image. Its selection directly influences the outcome of the inpainting or enhancement process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`dtype`**
    - Defines the data type for the BrushNet model's computations, affecting the precision and potentially the performance of the image processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale`**
    - A scaling factor that adjusts the intensity or effect of BrushNet's processing on the image. It allows for fine-tuning the impact of the modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Specifies the starting point for the processing within the image, allowing for targeted application of BrushNet's capabilities.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at`**
    - Defines the endpoint for the processing within the image, enabling precise control over the extent of BrushNet's modifications.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The modified pipeline context, including updated models and configurations, after BrushNet's processing has been applied to the image.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class applyBrushNet:

    def get_files_with_extension(folder='inpaint', extensions='.safetensors'):
        return [file for file in folder_paths.get_filename_list(folder) if file.endswith(extensions)]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "brushnet": (s.get_files_with_extension(),),
                "dtype": (['float16', 'bfloat16', 'float32', 'float64'], ),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "start_at": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    CATEGORY = "EasyUse/Inpaint"
    FUNCTION = "apply"

    def apply(self, pipe, image, mask, brushnet, dtype, scale, start_at, end_at):

        model = pipe['model']
        vae = pipe['vae']
        positive = pipe['positive']
        negative = pipe['negative']
        cls = BrushNet()
        if brushnet in backend_cache.cache:
            log_node_info("easy brushnetApply", f"Using {brushnet} Cached")
            _, brushnet_model = backend_cache.cache[brushnet][1]
        else:
            brushnet_file = os.path.join(folder_paths.get_full_path("inpaint", brushnet))
            brushnet_model, = cls.load_brushnet_model(brushnet_file, dtype)
            backend_cache.update_cache(brushnet, 'brushnet', (False, brushnet_model))
        m, positive, negative, latent = cls.brushnet_model_update(model=model, vae=vae, image=image, mask=mask,
                                                           brushnet=brushnet_model, positive=positive,
                                                           negative=negative, scale=scale, start_at=start_at,
                                                           end_at=end_at)
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
