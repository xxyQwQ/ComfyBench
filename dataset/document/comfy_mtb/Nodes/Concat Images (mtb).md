---
tags:
- Image
---

# Concat Images (mtb)
## Documentation
- Class name: `Concat Images (mtb)`
- Category: `mtb/image`
- Output node: `False`

The Concat Images node is designed to merge multiple images into a single batch, allowing for efficient processing and manipulation of image collections. It provides flexibility in handling size mismatches through customizable strategies.
## Input types
### Required
- **`reverse`**
    - Determines the order in which images are concatenated. When set to True, images are concatenated in reverse order.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`on_mismatch`**
    - Specifies the strategy to adopt when there is a size mismatch between images being concatenated. Options include 'Error', 'Smallest', and 'Largest', with 'Smallest' being the default.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single batched tensor containing all the concatenated images, ready for further processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_ConcatImages:
    """Add images to batch."""

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate_tensors"
    CATEGORY = "mtb/image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"reverse": ("BOOLEAN", {"default": False})},
            "optional": {
                "on_mismatch": (
                    ["Error", "Smallest", "Largest"],
                    {"default": "Smallest"},
                )
            },
        }

    def concatenate_tensors(
        self,
        reverse: bool,
        on_mismatch: str = "Smallest",
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        tensors = list(kwargs.values())

        if on_mismatch == "Error":
            shapes = [tensor.shape for tensor in tensors]
            if not all(shape == shapes[0] for shape in shapes):
                raise ValueError(
                    "All input tensors must have the same shape when on_mismatch is 'Error'."
                )

        else:
            import torch.nn.functional as F

            if on_mismatch == "Smallest":
                target_shape = min(
                    (tensor.shape for tensor in tensors),
                    key=lambda s: (s[1], s[2]),
                )
            else:  # on_mismatch == "Largest"
                target_shape = max(
                    (tensor.shape for tensor in tensors),
                    key=lambda s: (s[1], s[2]),
                )

            target_height, target_width = target_shape[1], target_shape[2]

            resized_tensors = []
            for tensor in tensors:
                if (
                    tensor.shape[1] != target_height
                    or tensor.shape[2] != target_width
                ):
                    resized_tensor = F.interpolate(
                        tensor.permute(0, 3, 1, 2),
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    resized_tensor = resized_tensor.permute(0, 2, 3, 1)
                    resized_tensors.append(resized_tensor)
                else:
                    resized_tensors.append(tensor)

            tensors = resized_tensors

        concatenated = torch.cat(tensors, dim=0)

        return (concatenated,)

```
