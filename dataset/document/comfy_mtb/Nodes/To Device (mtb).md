---
tags:
- Image
- ImageTransformation
---

# To Device (mtb)
## Documentation
- Class name: `To Device (mtb)`
- Category: `mtb/utils`
- Output node: `False`

The `MTB_ToDevice` node is designed to transfer image or mask tensors to a specified computing device, such as CPU, GPU, or MPS (Apple Silicon), enhancing computational efficiency and flexibility in data processing pipelines.
## Input types
### Required
- **`ignore_errors`**
    - Determines whether to proceed without throwing an error if both image and mask inputs are absent, allowing for more flexible error handling.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`device`**
    - Specifies the computing device to which the tensors will be transferred. It dynamically includes available options like CPU, GPU, and MPS, adapting to the system's capabilities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image`**
    - An optional image tensor that, if provided, will be transferred to the specified device.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor | None`
- **`mask`**
    - An optional mask tensor that, if provided, will be transferred to the specified device.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor | None`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The image tensor after being transferred to the specified device.
    - Python dtype: `torch.Tensor | None`
- **`masks`**
    - Comfy dtype: `MASK`
    - The mask tensor after being transferred to the specified device.
    - Python dtype: `torch.Tensor | None`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_ToDevice:
    """Send a image or mask tensor to the given device."""

    @classmethod
    def INPUT_TYPES(cls):
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda{i}")

        return {
            "required": {
                "ignore_errors": ("BOOLEAN", {"default": False}),
                "device": (devices, {"default": "cpu"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    CATEGORY = "mtb/utils"
    FUNCTION = "to_device"

    def to_device(
        self,
        *,
        ignore_errors=False,
        device="cuda",
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ):
        if not ignore_errors and image is None and mask is None:
            raise ValueError(
                "You must either provide an image or a mask,"
                " use ignore_error to passthrough"
            )
        if image is not None:
            image = image.to(device)
        if mask is not None:
            mask = mask.to(device)
        return (image, mask)

```
