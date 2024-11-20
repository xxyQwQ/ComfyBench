---
tags:
- Image
- ImageTransformation
---

# Stack Images (mtb)
## Documentation
- Class name: `Stack Images (mtb)`
- Category: `mtb/image utils`
- Output node: `False`

This node is designed to stack multiple input images either vertically or horizontally, based on the specified orientation. It ensures that all images are normalized to the same size and RGBA format before stacking, allowing for consistent and seamless image composition.
## Input types
### Required
- **`vertical`**
    - Determines the orientation of the stacking process. When true, images are stacked vertically, requiring uniform width across all images. When false, images are stacked horizontally, necessitating uniform height.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single image tensor that represents the stacked composition of the input images, arranged according to the specified orientation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_StackImages:
    """Stack the input images horizontally or vertically."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vertical": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stack"
    CATEGORY = "mtb/image utils"

    def stack(self, vertical, **kwargs):
        if not kwargs:
            raise ValueError("At least one tensor must be provided.")

        tensors = list(kwargs.values())
        log.debug(
            f"Stacking {len(tensors)} tensors "
            f"{'vertically' if vertical else 'horizontally'}"
        )

        normalized_tensors = [
            self.normalize_to_rgba(tensor) for tensor in tensors
        ]
        max_batch_size = max(tensor.shape[0] for tensor in normalized_tensors)
        normalized_tensors = [
            self.duplicate_frames(tensor, max_batch_size)
            for tensor in normalized_tensors
        ]

        if vertical:
            width = normalized_tensors[0].shape[2]
            if any(tensor.shape[2] != width for tensor in normalized_tensors):
                raise ValueError(
                    "All tensors must have the same width "
                    "for vertical stacking."
                )
            dim = 1
        else:
            height = normalized_tensors[0].shape[1]
            if any(tensor.shape[1] != height for tensor in normalized_tensors):
                raise ValueError(
                    "All tensors must have the same height "
                    "for horizontal stacking."
                )
            dim = 2

        stacked_tensor = torch.cat(normalized_tensors, dim=dim)

        return (stacked_tensor,)

    def normalize_to_rgba(self, tensor):
        """Normalize tensor to have 4 channels (RGBA)."""
        _, _, _, channels = tensor.shape
        # already RGBA
        if channels == 4:
            return tensor
        # RGB to RGBA
        elif channels == 3:
            alpha_channel = torch.ones(
                tensor.shape[:-1] + (1,), device=tensor.device
            )  # Add an alpha channel
            return torch.cat((tensor, alpha_channel), dim=-1)
        else:
            raise ValueError(
                "Tensor has an unsupported number of channels: "
                "expected 3 (RGB) or 4 (RGBA)."
            )

    def duplicate_frames(self, tensor, target_batch_size):
        """Duplicate frames in tensor to match the target batch size."""
        current_batch_size = tensor.shape[0]
        if current_batch_size < target_batch_size:
            duplication_factors: int = target_batch_size // current_batch_size
            duplicated_tensor = tensor.repeat(duplication_factors, 1, 1, 1)
            remaining_frames = target_batch_size % current_batch_size
            if remaining_frames > 0:
                duplicated_tensor = torch.cat(
                    (duplicated_tensor, tensor[:remaining_frames]), dim=0
                )
            return duplicated_tensor
        else:
            return tensor

```
