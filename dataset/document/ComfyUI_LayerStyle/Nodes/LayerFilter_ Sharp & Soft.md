---
tags:
- VisualEffects
---

# LayerFilter: Sharp & Soft
## Documentation
- Class name: `LayerFilter: Sharp & Soft`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The 'Sharp & Soft' node applies varying degrees of sharpness or softness to images, allowing for detailed enhancement or smoothing effects based on the selected mode. It dynamically adjusts filter parameters to achieve the desired level of image clarity or softness, enhancing visual quality or creating a more subtle, softened appearance.
## Input types
### Required
- **`images`**
    - The input images to be processed. This parameter is crucial as it directly influences the outcome by serving as the base for the sharpness or softness enhancement.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`enhance`**
    - Specifies the degree of sharpness or softness to apply to the images, ranging from 'very sharp' to 'very soft'. This choice determines the intensity of the effect and directly impacts the visual characteristics of the output images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The processed images, enhanced with the specified level of sharpness or softness. This output reflects the visual modifications applied to the input images, showcasing the effect of the chosen enhancement mode.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SharpAndSoft:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        enhance_list = ['very sharp', 'sharp', 'soft', 'very soft', 'None']

        return {
            "required": {
                "images": ("IMAGE",),
                "enhance": (enhance_list, ),

            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'sharp_and_soft'
    CATEGORY = '😺dzNodes/LayerFilter'

    def sharp_and_soft(self, images, enhance, ):

        if enhance == 'very sharp':
            filter_radius = 1
            denoise = 0.6
            detail_mult = 2.8
        if enhance == 'sharp':
            filter_radius = 3
            denoise = 0.12
            detail_mult = 1.8
        if enhance == 'soft':
            filter_radius = 8
            denoise = 0.08
            detail_mult = 0.5
        if enhance == 'very soft':
            filter_radius = 15
            denoise = 0.06
            detail_mult = 0.01
        else:
            return (images,)

        d = int(filter_radius * 2) + 1
        s = 0.02
        n = denoise / 10
        dup = copy.deepcopy(images.cpu().numpy())

        for index, image in enumerate(dup):
            imgB = image
            if denoise > 0.0:
                imgB = cv2.bilateralFilter(image, d, n, d)
            imgG = np.clip(guidedFilter(image, image, d, s), 0.001, 1)
            details = (imgB / imgG - 1) * detail_mult + 1
            dup[index] = np.clip(details * imgG - imgB + image, 0, 1)

        log(f"{NODE_NAME} Processed {dup.shape[0]} image(s).", message_type='finish')
        return (torch.from_numpy(dup),)

```
