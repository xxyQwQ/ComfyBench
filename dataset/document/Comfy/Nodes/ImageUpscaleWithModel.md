---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# Upscale Image (using Model)
## Documentation
- Class name: `ImageUpscaleWithModel`
- Category: `image/upscaling`
- Output node: `False`

This node is designed to upscale images using a specified upscale model. It dynamically manages memory requirements based on the model and image size, performs the upscaling in a tiled manner to handle large images efficiently, and ensures the output image is clamped within a valid range.
## Input types
### Required
- **`upscale_model`**
    - The upscale model to be used for upscaling the image. It determines the upscaling algorithm and its parameters.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `torch.nn.Module`
- **`image`**
    - The input image to be upscaled. The image is processed and upscaled according to the specified upscale model.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The upscaled image, with pixel values clamped to the range [0, 1].
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImageScaleBy](../../Comfy/Nodes/ImageScaleBy.md)
    - [ImageScale](../../Comfy/Nodes/ImageScale.md)
    - [SaveImage](../../Comfy/Nodes/SaveImage.md)
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - Reroute
    - [ImageScaleToTotalPixels](../../Comfy/Nodes/ImageScaleToTotalPixels.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [CR Image Output](../../ComfyUI_Comfyroll_CustomNodes/Nodes/CR Image Output.md)
    - [ImageCASharpening+](../../ComfyUI_essentials/Nodes/ImageCASharpening+.md)



## Source code
```python
class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)

```
