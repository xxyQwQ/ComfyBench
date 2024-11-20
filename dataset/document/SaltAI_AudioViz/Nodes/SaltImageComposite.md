---
tags:
- Image
- ImageBlend
- ImageComposite
---

# Scheduled Image Composite
## Documentation
- Class name: `SaltImageComposite`
- Category: `SALT/AudioViz/Scheduling/Image`
- Output node: `False`

This node is designed for creating complex image compositions by blending multiple images together according to specified parameters and schedules. It enables the dynamic assembly of visual elements, allowing for the creation of rich, layered imagery that can be adjusted over time or in response to specific triggers.
## Input types
### Required
- **`images_a`**
    - Specifies the first set of images to be blended. It plays a crucial role in determining the base layer of the composition.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`images_b`**
    - Specifies the second set of images to be blended with the first set. This layer is combined with the base layer according to the blending mode and other parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`mode`**
    - Defines the blending mode to be used for combining the images. The choice of mode affects the visual outcome of the blend, offering various stylistic effects.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`masks`**
    - Optional masks that can be applied to the images for selective blending, allowing for more controlled and intricate compositions.
    - Comfy dtype: `MASK`
    - Python dtype: `List[torch.Tensor]`
- **`blend_schedule`**
    - An optional schedule that defines how the blend ratio changes over time, enabling dynamic transitions between images.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting images after blending, showcasing the combined visual elements from the input images.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltImageComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "mode": ([
                    "add",
                    "color",
                    "color_burn",
                    "color_dodge",
                    "darken",
                    "difference",
                    "exclusion",
                    "hard_light",
                    "hue",
                    "lighten",
                    "multiply",
                    "overlay",
                    "screen",
                    "soft_light"
                ],),
            },
            "optional": {
                "masks": ("MASK",),
                "blend_schedule": ("LIST", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "blend"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Image"

    def blend(self, images_a, images_b, mode, blend_schedule=[1.0], masks=None):
        blended_images = []
        min_length = min(len(images_a), len(images_b))

        if len(blend_schedule) < min_length:
            blend_schedule += [blend_schedule[-1]] * (min_length - len(blend_schedule))

        for i in range(min_length):
            img_a = tensor2pil(images_a[i].unsqueeze(0))
            img_b = tensor2pil(images_b[i].unsqueeze(0))
            img_b_resized = img_b.resize(img_a.size, Image.LANCZOS).convert(img_a.mode)

            if mode == "add":
                base_image = ImageChops.add(img_a, img_b_resized, scale=2.0, offset=int(255 * (1 - blend_schedule[i])))
            else:
                base_image = getattr(pilgram.css.blending, mode)(img_a, img_b_resized)

            blend_mask = Image.new("L", img_a.size, int(255 * blend_schedule[i]))
            out_image = Image.composite(base_image, img_a, blend_mask)

            if isinstance(masks, torch.Tensor):
                mask = mask2pil(masks[i if len(masks) > i else -1].unsqueeze(0)).resize(img_a.size, Image.LANCZOS).convert("L")
                final_image = Image.composite(out_image, img_a, mask)
            else:
                final_image = out_image

            blended_images.append(pil2tensor(final_image))

        blended_images_batch = torch.cat(blended_images, dim=0)
        return (blended_images_batch, )

```
