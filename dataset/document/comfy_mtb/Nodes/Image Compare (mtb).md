---
tags:
- Comparison
---

# Image Compare (mtb)
## Documentation
- Class name: `Image Compare (mtb)`
- Category: `mtb/image`
- Output node: `False`

This node compares two images using different modes such as difference, blend, or checkerboard, and returns a new image highlighting the differences or combinations based on the selected mode.
## Input types
### Required
- **`imageA`**
    - The first image to compare, represented as a torch.Tensor. It is one of the primary inputs for the comparison operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`imageB`**
    - The second image to compare, also represented as a torch.Tensor. This image is compared against the first to identify differences or create a blend/checkerboard effect.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - A string indicating the comparison mode: 'checkerboard', 'diff', or 'blend'. The mode determines how the two images are compared and combined.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after comparison, highlighting differences or combinations based on the selected mode.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MTB_ImageCompare:
    """Compare two images and return a difference image"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "imageA": ("IMAGE",),
                "imageB": ("IMAGE",),
                "mode": (
                    ["checkerboard", "diff", "blend"],
                    {"default": "checkerboard"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compare"
    CATEGORY = "mtb/image"

    def compare(self, imageA: torch.Tensor, imageB: torch.Tensor, mode):
        if imageA.dim() == 4:
            batch_count = imageA.size(0)
            return (
                torch.cat(
                    tuple(
                        self.compare(imageA[i], imageB[i], mode)[0]
                        for i in range(batch_count)
                    ),
                    dim=0,
                ),
            )

        num_channels_A = imageA.size(2)
        num_channels_B = imageB.size(2)

        # handle RGBA/RGB mismatch
        if num_channels_A == 3 and num_channels_B == 4:
            imageA = torch.cat(
                (imageA, torch.ones_like(imageA[:, :, 0:1])), dim=2
            )
        elif num_channels_B == 3 and num_channels_A == 4:
            imageB = torch.cat(
                (imageB, torch.ones_like(imageB[:, :, 0:1])), dim=2
            )
        match mode:
            case "diff":
                compare_image = torch.abs(imageA - imageB)
            case "blend":
                compare_image = 0.5 * (imageA + imageB)
            case "checkerboard":
                imageA = imageA.numpy()
                imageB = imageB.numpy()
                compared_channels = [
                    torch.from_numpy(
                        compare_images(
                            imageA[:, :, i], imageB[:, :, i], method=mode
                        )
                    )
                    for i in range(imageA.shape[2])
                ]

                compare_image = torch.stack(compared_channels, dim=2)
            case _:
                compare_image = None
                raise ValueError(f"Unknown mode {mode}")

        compare_image = compare_image.unsqueeze(0)

        return (compare_image,)

```
