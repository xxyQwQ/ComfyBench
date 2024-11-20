---
tags:
- Scheduling
---

# Scheduled Binary Comparison
## Documentation
- Class name: `SaltScheduledBinaryComparison`
- Category: `SALT/AudioViz/Scheduling/Image`
- Output node: `False`

This node performs scheduled binary comparisons on a batch of images, applying a dynamic thresholding technique that can optionally incorporate a margin of error (epsilon) to determine the binary outcome. It is designed to process images in batches, adjusting the comparison threshold per image based on a provided schedule, thereby enabling complex, time-varying binary image transformations.
## Input types
### Required
- **`images`**
    - The batch of images to be processed. This parameter is crucial for defining the input data on which the binary comparison and thresholding operations will be performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`comparison_schedule`**
    - A schedule of comparison threshold values to be applied to each image in the batch. This parameter allows for dynamic adjustment of the thresholding criteria over the batch.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
### Optional
- **`epsilon_schedule`**
    - An optional schedule of epsilon values providing a margin of error for the comparison, enabling a more flexible thresholding operation.
    - Comfy dtype: `LIST`
    - Python dtype: `List[float]`
- **`use_epsilon`**
    - A flag to determine whether the epsilon margin of error should be used in the comparison, offering a toggle between strict and flexible thresholding.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting batch of images after applying the scheduled binary comparison and thresholding operations, represented as binary outcomes.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltScheduledBinaryComparison:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "comparison_schedule": ("LIST",),
            },
            "optional": {
                "epsilon_schedule": ("LIST", {}),
                "use_epsilon": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    FUNCTION = "binary_threshold"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Scheduling/Image"

    def binary_threshold(self, images, comparison_schedule, epsilon_schedule=[0.1], use_epsilon=True):
        batch_size = images.shape[0]

        if len(comparison_schedule) < batch_size:
            last_val = comparison_schedule[-1]
            comparison_schedule.extend([last_val] * (batch_size - len(comparison_schedule)))
        comparison_schedule = comparison_schedule[:batch_size]

        thresholds_tensor = torch.tensor(comparison_schedule, dtype=images.dtype).view(batch_size, 1, 1, 1)
        
        if use_epsilon:
            if epsilon_schedule is None:
                epsilon_schedule = [0.1] * batch_size
            if len(epsilon_schedule) < batch_size:
                last_eps = epsilon_schedule[-1]
                epsilon_schedule.extend([last_eps] * (batch_size - len(epsilon_schedule)))
            epsilon_schedule = epsilon_schedule[:batch_size]
            epsilon_tensor = torch.tensor(epsilon_schedule, dtype=images.dtype).view(batch_size, 1, 1, 1)
            
            condition_met = ((images == thresholds_tensor) |
                             (torch.abs(images - thresholds_tensor) <= epsilon_tensor))
        else:
            condition_met = images >= thresholds_tensor

        thresholded_images = torch.where(condition_met, torch.tensor(1.0, dtype=images.dtype), torch.tensor(0.0, dtype=images.dtype))

        return (thresholded_images, )

```
