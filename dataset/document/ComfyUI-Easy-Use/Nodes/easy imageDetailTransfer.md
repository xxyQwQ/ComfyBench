---
tags:
- VisualEffects
---

# Image Detail Transfer
## Documentation
- Class name: `easy imageDetailTransfer`
- Category: `EasyUse/Image`
- Output node: `True`

This node specializes in transferring intricate details from a source image to a target image, enhancing the target with characteristics or styles found in the source. It operates by blending visual elements between images, potentially using various modes of blending to achieve the desired effect. The node is designed for applications requiring high fidelity in detail preservation or stylistic amalgamation, making it suitable for tasks like photo editing, artistic creation, and visual content enhancement.
## Input types
### Required
- **`target`**
    - The 'target' parameter represents the image that will receive the details or styles from the source image. It plays a crucial role in the operation by serving as the canvas on which the transferred details are applied, directly influencing the outcome of the detail transfer process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`source`**
    - The 'source' parameter specifies the image from which details or styles will be extracted and transferred to the target image. Its content significantly affects the detail transfer process, as it provides the visual elements to be blended into the target.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - The 'mode' parameter determines the blending technique used to transfer details from the source to the target image. It influences the visual outcome of the transfer, offering various options for how the details are integrated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`blur_sigma`**
    - Specifies the standard deviation for the Gaussian blur applied to the target image, affecting the smoothness of the detail transfer.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blend_factor`**
    - Controls the intensity of the blending between the source and target images, directly influencing the visibility of transferred details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`image_output`**
    - Determines how the output image is handled, whether it is hidden, previewed, saved, or a combination of these actions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - A prefix added to the filename when saving the output image, helping in organizing and identifying the results.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`mask`**
    - An optional mask that can be applied to the target image, specifying areas where the detail transfer should or should not occur. This parameter allows for more precise control over the application of transferred details, enabling selective enhancement based on the mask's coverage.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final image after the detail transfer process, incorporating the blended details from the source image into the target.
    - Python dtype: `torch.Tensor`
- **`ui`**
    - The 'ui' parameter contains the results of the image detail transfer process, showcasing the enhanced target image with details transferred from the source. It serves as a visual confirmation of the operation's success.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageDetailTransfer:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "target": ("IMAGE",),
        "source": ("IMAGE",),
        "mode": (["add", "multiply", "screen", "overlay", "soft_light", "hard_light", "color_dodge", "color_burn", "difference", "exclusion", "divide",],{"default": "add"}),
        "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
        "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001, "round": 0.001}),
        "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
        "save_prefix": ("STRING", {"default": "ComfyUI"}),
      },
      "optional": {
        "mask": ("MASK",),
      },
      "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  OUTPUT_NODE = True
  FUNCTION = "transfer"
  CATEGORY = "EasyUse/Image"



  def transfer(self, target, source, mode, blur_sigma, blend_factor, image_output, save_prefix, mask=None, prompt=None, extra_pnginfo=None):
    batch_size, height, width, _ = target.shape
    device = comfy.model_management.get_torch_device()
    target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
    source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

    if target.shape[1:] != source.shape[1:]:
      source_tensor = comfy.utils.common_upscale(source_tensor, width, height, "bilinear", "disabled")

    if source.shape[0] < batch_size:
      source = source[0].unsqueeze(0).repeat(batch_size, 1, 1, 1)

    kernel_size = int(6 * int(blur_sigma) + 1)

    gaussian_blur = GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)

    if mode == "add":
      new_image = (source_tensor - blurred_source) + blurred_target
    elif mode == "multiply":
      new_image = source_tensor * blurred_target
    elif mode == "screen":
      new_image = 1 - (1 - source_tensor) * (1 - blurred_target)
    elif mode == "overlay":
      new_image = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "soft_light":
      new_image = (1 - 2 * blurred_target) * source_tensor ** 2 + 2 * blurred_target * source_tensor
    elif mode == "hard_light":
      new_image = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target,
                               1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "difference":
      new_image = torch.abs(blurred_target - source_tensor)
    elif mode == "exclusion":
      new_image = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
    elif mode == "color_dodge":
      new_image = blurred_target / (1 - source_tensor)
    elif mode == "color_burn":
      new_image = 1 - (1 - blurred_target) / source_tensor
    elif mode == "divide":
      new_image = (source_tensor / blurred_source) * blurred_target
    else:
      new_image = source_tensor

    new_image = torch.lerp(target_tensor, new_image, blend_factor)
    if mask is not None:
      mask = mask.to(device)
      new_image = torch.lerp(target_tensor, new_image, mask)
    new_image = torch.clamp(new_image, 0, 1)
    new_image = new_image.permute(0, 2, 3, 1).cpu().float()

    results = easySave(new_image, save_prefix, image_output, prompt, extra_pnginfo)

    if image_output in ("Hide", "Hide/Save"):
      return {"ui": {},
              "result": (new_image,)}

    return {"ui": {"images": results},
            "result": (new_image,)}

```
