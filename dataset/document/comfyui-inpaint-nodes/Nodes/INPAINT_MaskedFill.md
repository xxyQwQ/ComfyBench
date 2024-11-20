---
tags:
- Mask
---

# Fill Masked Area
## Documentation
- Class name: `INPAINT_MaskedFill`
- Category: `inpaint`
- Output node: `False`

The 'Fill Masked Area' node is designed to fill areas of an image that are masked, using either a neutral color fill or advanced inpainting techniques. It provides options for different fill methods, including a simple neutral fill or more complex methods leveraging OpenCV's inpainting algorithms, to seamlessly blend the filled area with the surrounding image content.
## Input types
### Required
- **`image`**
    - The image tensor to be processed, where masked areas are to be filled. It serves as the primary input for the inpainting operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - A tensor representing the mask, indicating areas of the image to be filled. It plays a crucial role in determining which parts of the image are subject to inpainting.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`fill`**
    - Specifies the method of filling: 'neutral' for a simple color fill, or 'telea'/'ns' for using OpenCV's inpainting algorithms. This choice affects the visual outcome of the filled areas.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`falloff`**
    - An integer defining the falloff effect around the edges of the mask, enhancing the blending of filled areas with the rest of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image tensor after the masked areas have been filled, showcasing the result of the inpainting operation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskedFill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "fill": (["neutral", "telea", "navier-stokes"],),
                "falloff": ("INT", {"default": 0, "min": 0, "max": 8191, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "inpaint"
    FUNCTION = "fill"

    def fill(self, image: Tensor, mask: Tensor, fill: str, falloff: int):
        image = image.detach().clone()
        alpha = mask_unsqueeze(mask_floor(mask))
        assert alpha.shape[0] == image.shape[0], "Image and mask batch size does not match"

        falloff = make_odd(falloff)
        if falloff > 0:
            erosion = binary_erosion(alpha, falloff)
            alpha = alpha * gaussian_blur(erosion, falloff)

        if fill == "neutral":
            m = (1.0 - alpha).squeeze(1)
            for i in range(3):
                image[:, :, :, i] -= 0.5
                image[:, :, :, i] *= m
                image[:, :, :, i] += 0.5
        else:
            import cv2

            method = cv2.INPAINT_TELEA if fill == "telea" else cv2.INPAINT_NS
            for slice, alpha_slice in zip(image, alpha):
                alpha_np = alpha_slice.squeeze().cpu().numpy()
                alpha_bc = alpha_np.reshape(*alpha_np.shape, 1)
                image_np = slice.cpu().numpy()
                filled_np = cv2.inpaint(
                    (255.0 * image_np).astype(np.uint8),
                    (255.0 * alpha_np).astype(np.uint8),
                    3,
                    method,
                )
                filled_np = filled_np.astype(np.float32) / 255.0
                filled_np = image_np * (1.0 - alpha_bc) + filled_np * alpha_bc
                slice.copy_(torch.from_numpy(filled_np))

        return (image,)

```
