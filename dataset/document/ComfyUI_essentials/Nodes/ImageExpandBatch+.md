---
tags:
- Batch
- Image
- ImageBatch
---

# 🔧 Image Expand Batch
## Documentation
- Class name: `ImageExpandBatch+`
- Category: `essentials/image batch`
- Output node: `False`

This node is designed to facilitate the manipulation of image batches within a graphical interface, specifically focusing on expanding a given batch of images. It abstracts the complexities involved in handling multiple images simultaneously, providing a streamlined approach to either augment the existing batch size or modify the batch in a way that accommodates additional image processing operations.
## Input types
### Required
- **`image`**
    - The primary image or batch of images to be expanded. This parameter is the basis for the expansion operation, determining the initial set of images to be modified or augmented.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`size`**
    - Specifies the target size for the batch expansion. This could dictate the number of times the image(s) are repeated or the new size of the batch after expansion.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`method`**
    - Defines the method of expansion, such as repeating the entire batch, repeating only the first or last image, or expanding the batch size in another specified manner.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an expanded batch of images, modified according to the specified size and method. This facilitates further batch-level image processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image, size, method):
        orig_size = image.shape[0]

        if orig_size == size:
            return (image,)

        if size <= 1:
            return (image[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = image[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = image[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = image.repeat([math.ceil(size / image.shape[0])] + [1] * (len(image.shape) - 1))[:size]
        elif 'first' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat([image[:1].repeat(size-image.shape[0], 1, 1, 1), image], dim=0)
        elif 'last' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat((image, image[-1:].repeat((size-image.shape[0], 1, 1, 1))), dim=0)

        return (out,)

```
