---
tags:
- Mask
---

# LayerMask: MaskByDifferent
## Documentation
- Class name: `LayerMask: MaskByDifferent`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node is designed to create a mask based on differences between two images, enabling the identification and isolation of unique elements. It facilitates dynamic segmentation by analyzing and contrasting the visual characteristics present in each layer.
## Input types
### Required
- **`image_i`**
    - The first and second images to be compared, serving as reference points for identifying differences. These images are used to detect variations and unique features, enabling the creation of a mask that highlights these differences.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`gain`**
    - Adjusts the sensitivity of the difference detection, influencing the intensity of the identified changes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`fix_gap`**
    - Defines the tolerance for gap fixing in the generated mask, helping to smooth and refine the mask's edges.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_threshold`**
    - Sets the threshold for fixing gaps within the mask, further enhancing mask accuracy by refining edge details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`main_subject_detect`**
    - A boolean flag that, when enabled, focuses the difference detection on the main subject of the images, potentially improving the relevance of the generated mask.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The resulting mask that highlights the differences between the two input images, isolating unique elements and features.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskByDifferent:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image_1": ("IMAGE", ),  #
                "image_2": ("IMAGE",),  #
                "gain": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 100, "step": 0.1}),
                "fix_gap": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
                "fix_threshold": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}),
                "main_subject_detect": ("BOOLEAN", {"default": False}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ( "MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_by_different'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_by_different(self, image_1, image_2, gain, fix_gap, fix_threshold, main_subject_detect):

        image1s = []
        image2s = []
        ret_masks = []
        for i in image_1:
            image1s.append(torch.unsqueeze(i, 0))
        for i in image_2:
            image2s.append(torch.unsqueeze(i, 0))
        max_batch = max(len(image1s), len(image2s))
        blank_mask = image2mask(Image.new('L', size=tensor2pil(image1s[0]).size, color='black'))
        if tensor2pil(image1s[0]).size != tensor2pil(image2s[0]).size:
            log(f"Error: {NODE_NAME} skipped, because the image size is not match.", message_type='error')
            return (torch.cat([blank_mask], dim=0))
        for i in range(max_batch):
            t1 = image1s[i] if i < len(image1s) else image1s[-1]
            t2 = image2s[i] if i < len(image2s) else image2s[-1]
            t1 = pil2tensor(tensor2pil(t1).convert('RGB'))
            t2 = pil2tensor(tensor2pil(t2).convert('RGB'))
            t = torch.abs(t1 - t2) * gain
            _mask = mask_fix(t, 1, fix_gap, fix_threshold, fix_threshold)
            _mask = tensor2pil(_mask)
            if main_subject_detect:
                subject_mask1 = RMBG(tensor2pil(t1))
                subject_mask2 = RMBG(tensor2pil(t2))
                subject_mask = chop_image(subject_mask1, subject_mask2, blend_mode='add', opacity=100)
                grow = (subject_mask.width + subject_mask.height) // 100
                subject_mask = mask2image(expand_mask(image2mask(subject_mask), grow * 2, grow))
                black = Image.new('L', size=_mask.size, color='black')
                white = Image.new('L', size=_mask.size, color='white')
                black.paste(_mask, mask=subject_mask.convert('L'))
                black.paste(white, mask=subject_mask1.convert('L'))
                black.paste(white, mask=subject_mask2.convert('L'))
                _mask = black

            ret_masks.append(image2mask(_mask))

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
