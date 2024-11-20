---
tags:
- Image
---

# 🔧 Image Seam Carving
## Documentation
- Class name: `ImageSeamCarving+`
- Category: `essentials/image manipulation`
- Output node: `False`

This node specializes in dynamically resizing images while preserving the content's essential features, utilizing the seam carving technique. It intelligently removes or adds pixels in paths (seams) of least importance, allowing for aspect ratio changes, content amplification, or object removal without noticeable distortion.
## Input types
### Required
- **`image`**
    - The input image to be processed. It is the primary subject for seam carving, where the algorithm will apply modifications based on the specified parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - The target width for the output image. This parameter dictates the new width the input image should be resized to, either by removing or adding seams.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The target height for the output image. Similar to width, it determines the new height by adjusting the number of seams.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`energy`**
    - Defines the energy calculation mode to identify seams. Higher energy values indicate areas of higher importance, guiding the seam carving process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`order`**
    - Specifies the order of seam removal or addition, which can be either width-first or height-first, influencing the direction of the carving process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`keep_mask`**
    - An optional mask to protect certain areas from being altered. Regions covered by the keep mask are treated as high importance, preserving them during seam carving.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`drop_mask`**
    - An optional mask to designate areas for removal. The drop mask marks regions as less important, making them prime candidates for seam elimination.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after seam carving. It reflects the specified dimensions and modifications while aiming to preserve the visual content's integrity.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageSeamCarving:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1, }),
                "energy": (["backward", "forward"],),
                "order": (["width-first", "height-first"],),
            },
            "optional": {
                "keep_mask": ("MASK",),
                "drop_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "essentials/image manipulation"
    FUNCTION = "execute"

    def execute(self, image, width, height, energy, order, keep_mask=None, drop_mask=None):
        from .carve import seam_carving

        img = image.permute([0, 3, 1, 2])

        if keep_mask is not None:
            #keep_mask = keep_mask.reshape((-1, 1, keep_mask.shape[-2], keep_mask.shape[-1])).movedim(1, -1)
            keep_mask = keep_mask.unsqueeze(1)

            if keep_mask.shape[2] != img.shape[2] or keep_mask.shape[3] != img.shape[3]:
                keep_mask = F.interpolate(keep_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")
        if drop_mask is not None:
            drop_mask = drop_mask.unsqueeze(1)

            if drop_mask.shape[2] != img.shape[2] or drop_mask.shape[3] != img.shape[3]:
                drop_mask = F.interpolate(drop_mask, size=(img.shape[2], img.shape[3]), mode="bilinear")

        out = []
        for i in range(img.shape[0]):
            resized = seam_carving(
                T.ToPILImage()(img[i]),
                size=(width, height),
                energy_mode=energy,
                order=order,
                keep_mask=T.ToPILImage()(keep_mask[i]) if keep_mask is not None else None,
                drop_mask=T.ToPILImage()(drop_mask[i]) if drop_mask is not None else None,
            )
            out.append(T.ToTensor()(resized))

        out = torch.stack(out).permute([0, 2, 3, 1])

        return(out, )

```
