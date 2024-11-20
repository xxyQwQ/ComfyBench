---
tags:
- VisualEffects
---

# Detail Transfer
## Documentation
- Class name: `DetailTransfer`
- Category: `IC-Light`
- Output node: `False`

The DetailTransfer node is designed to enhance the visual quality of images by transferring intricate details from one image to another. It focuses on improving the texture and depth of the target image, ensuring that the final output retains a high level of detail and realism.
## Input types
### Required
- **`target`**
    - The target image that will receive the detail enhancements, serving as the primary canvas for the detail transfer.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`source`**
    - The source image from which details are extracted to be transferred to the target image, playing a key role in defining the transferred textures and details.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the blending mode used to combine the source and target images, influencing the final appearance of the detail transfer.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`blur_sigma`**
    - Determines the amount of blur applied to the detail transfer, affecting the smoothness and subtlety of the detail integration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blend_factor`**
    - Controls the intensity of the detail transfer, allowing for fine-tuning of how prominently the transferred details appear on the target image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask`**
    - An optional mask that can specify areas of the target image to be affected or excluded from the detail transfer, offering additional control over the outcome.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced image with transferred details, showcasing improved texture and depth for a more realistic and detailed appearance.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DetailTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target": ("IMAGE", ),
                "source": ("IMAGE", ),
                "mode": ([
                    "add",
                    "multiply",
                    "screen",
                    "overlay",
                    "soft_light",
                    "hard_light",
                    "color_dodge",
                    "color_burn",
                    "difference",
                    "exclusion",
                    "divide",
                    
                    ], 
                    {"default": "add"}
                    ),
                "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "IC-Light"

    def adjust_mask(self, mask, target_tensor):
        # Add a channel dimension and repeat to match the channel number of the target tensor
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # Add a channel dimension
            target_channels = target_tensor.shape[1]
            mask = mask.expand(-1, target_channels, -1, -1)  # Expand the channel dimension to match the target tensor's channels
    
        return mask


    def process(self, target, source, mode, blur_sigma, blend_factor, mask=None):
        B, H, W, C = target.shape
        device = model_management.get_torch_device()
        target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
        source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

        if target.shape[1:] != source.shape[1:]:
            source_tensor = comfy.utils.common_upscale(source_tensor, W, H, "bilinear", "disabled")

        if source.shape[0] < B:
            source = source[0].unsqueeze(0).repeat(B, 1, 1, 1)

        kernel_size = int(6 * int(blur_sigma) + 1)

        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

        blurred_target = gaussian_blur(target_tensor)
        blurred_source = gaussian_blur(source_tensor)
        
        if mode == "add":
            tensor_out = (source_tensor - blurred_source) + blurred_target
        elif mode == "multiply":
            tensor_out = source_tensor * blurred_target
        elif mode == "screen":
            tensor_out = 1 - (1 - source_tensor) * (1 - blurred_target)
        elif mode == "overlay":
            tensor_out = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
        elif mode == "soft_light":
            tensor_out = (1 - 2 * blurred_target) * source_tensor**2 + 2 * blurred_target * source_tensor
        elif mode == "hard_light":
            tensor_out = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
        elif mode == "difference":
            tensor_out = torch.abs(blurred_target - source_tensor)
        elif mode == "exclusion":
            tensor_out = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
        elif mode == "color_dodge":
            tensor_out = blurred_target / (1 - source_tensor)
        elif mode == "color_burn":
            tensor_out = 1 - (1 - blurred_target) / source_tensor
        elif mode == "divide":
            tensor_out = (source_tensor / blurred_source) * blurred_target
        else:
            tensor_out = source_tensor
        
        tensor_out = torch.lerp(target_tensor, tensor_out, blend_factor)
        if mask is not None:
            # Call the function and pass in mask and target_tensor
            mask = self.adjust_mask(mask, target_tensor)
            mask = mask.to(device)
            tensor_out = torch.lerp(target_tensor, tensor_out, mask)
        tensor_out = torch.clamp(tensor_out, 0, 1)
        tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()
        return (tensor_out,)

```
