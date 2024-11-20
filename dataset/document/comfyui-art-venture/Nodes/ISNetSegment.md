---
tags:
- Segmentation
---

# ISNet Segment
## Documentation
- Class name: `ISNetSegment`
- Category: `Art Venture/Segmentation`
- Output node: `False`

ISNetSegment is designed for image segmentation using the ISNet model, providing functionality to process images through the model to generate segmented images and their corresponding masks. It supports conditional execution based on the model's availability and the option to specify device mode for computation.
## Input types
### Required
- **`images`**
    - The input images to be segmented. This is the primary data upon which segmentation is performed, determining the visual content to be analyzed and processed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`threshold`**
    - A threshold value for segmentation, influencing the sensitivity of the segmentation process. It determines the cut-off point for what is considered part of the segment versus the background.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`device_mode`**
    - Specifies the computational device preference ('AUTO', 'Prefer GPU', 'CPU') for running the segmentation, affecting performance and resource utilization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`enabled`**
    - A boolean flag to enable or disable the segmentation process, allowing for conditional execution based on external criteria.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`isnet_model`**
    - An optional ISNet model instance to be used for segmentation. If not provided, the system will attempt to load a default model.
    - Comfy dtype: `ISNET_MODEL`
    - Python dtype: `ISNetBase`
## Output types
- **`segmented`**
    - Comfy dtype: `IMAGE`
    - The segmented images resulting from the application of the ISNet model on the input images.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The masks corresponding to the segmented images, indicating the segmented areas.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ISNetSegment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.001}),
            },
            "optional": {
                "device_mode": (["AUTO", "Prefer GPU", "CPU"],),
                "enabled": ("BOOLEAN", {"default": True}),
                "isnet_model": ("ISNET_MODEL",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("segmented", "mask")
    CATEGORY = "Art Venture/Segmentation"
    FUNCTION = "segment_isnet"

    def segment_isnet(self, images: torch.Tensor, threshold, device_mode="AUTO", enabled=True, isnet_model=None):
        if not enabled:
            masks = torch.zeros((len(images), 64, 64), dtype=torch.float32)
            return (images, masks)

        if isnet_model is None:
            ckpts = folder_paths.get_filename_list("isnet")
            if len(ckpts) == 0:
                ckpts = download_model(
                    model_path=model_dir,
                    model_url=model_url,
                    ext_filter=[".pth"],
                    download_name="isnet-general-use.pth",
                )
            isnet_model = load_isnet_model(ckpts[0])

        device = gpu if device_mode != "CPU" else cpu
        isnet_model = isnet_model.to(device)

        try:
            segments = []
            masks = []
            for image in images:
                mask = predict(isnet_model, image, device)
                mask_im = tensor2pil(mask.permute(1, 2, 0))
                cropped = Image.new("RGBA", mask_im.size, (0,0,0,0))
                cropped.paste(tensor2pil(image), mask=mask_im)

                masks.append(mask)
                segments.append(pil2tensor(cropped))

            return (torch.cat(segments, dim=0), torch.stack(masks))
        finally:
            if device_mode == "AUTO":
                isnet_model = isnet_model.to(cpu)

```
