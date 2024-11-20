---
tags:
- DepthMap
- Image
---

# Image Injection 🎭🅐🅓
## Documentation
- Class name: `ADE_NoisedImageInjection`
- Category: `Animate Diff 🎭🅐🅓/sample settings/image inject`
- Output node: `False`

This node is designed to facilitate the injection of noised images into a generative model's sampling process. It enables the dynamic alteration of images by applying noise and other transformations based on specified parameters, thereby enhancing the diversity and creativity of the generated outputs. The node's functionality is pivotal for scenarios requiring the integration of pre-existing imagery with new generative elements, offering a blend of precision and artistic freedom in the creation process.
## Input types
### Required
- **`image`**
    - The original image to be injected with noise and potentially other transformations. It serves as the base for the injection process, influencing the final output by providing the initial visual context.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - A variational autoencoder used for encoding and decoding images during the injection process. It plays a crucial role in ensuring that the injected images are compatible with the model's latent space, facilitating seamless integration.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
### Optional
- **`mask_opt`**
    - An optional mask that specifies areas of the image to be affected by the injection process. It allows for targeted alterations, enhancing the precision of the injection.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor or None`
- **`invert_mask`**
    - A boolean flag that, when true, inverts the mask's effect, targeting the injection process to areas outside the mask.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`resize_image`**
    - A boolean flag indicating whether the image should be resized as part of the injection process. This can be crucial for matching the dimensions expected by the model.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`start_percent`**
    - Specifies the starting point of the injection process as a percentage of the total steps, allowing for timed or phased injections.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`guarantee_steps`**
    - The minimum number of steps for which the injection will be applied, ensuring a certain level of impact from the injection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`img_inject_opts`**
    - Options for fine-tuning the injection process, such as positioning of the injected image.
    - Comfy dtype: `IMAGE_INJECT_OPTIONS`
    - Python dtype: `NoisedImageInjectOptions or None`
- **`strength_multival`**
    - A multiplier for the injection strength, allowing for adjustments in the intensity of the applied noise or transformations.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `float or torch.Tensor or None`
- **`prev_image_inject`**
    - The previous group of injected images, enabling sequential and cumulative injections over multiple steps.
    - Comfy dtype: `IMAGE_INJECT`
    - Python dtype: `NoisedImageToInjectGroup or None`
## Output types
- **`image_inject`**
    - Comfy dtype: `IMAGE_INJECT`
    - The updated group of injected images, including the newly injected image. This output facilitates sequential image injections, allowing for cumulative alterations over multiple steps.
    - Python dtype: `NoisedImageToInjectGroup`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class NoisedImageInjectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": ("VAE", ),
            },
            "optional": {
                "mask_opt": ("MASK", ),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "resize_image": ("BOOLEAN", {"default": True}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "guarantee_steps": ("INT", {"default": 1, "min": 1, "max": BIGMAX}),
                "img_inject_opts": ("IMAGE_INJECT_OPTIONS", ),
                "strength_multival": ("MULTIVAL", ),
                "prev_image_inject": ("IMAGE_INJECT", ),
            }
        }
    
    RETURN_TYPES = ("IMAGE_INJECT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/sample settings/image inject"
    FUNCTION = "create_image_inject"

    def create_image_inject(self, image: Tensor, vae: VAE, invert_mask: bool, resize_image: bool, start_percent: float,
                            mask_opt: Tensor=None, strength_multival: Union[float, Tensor]=None, prev_image_inject: NoisedImageToInjectGroup=None, guarantee_steps=1,
                            img_inject_opts=None):
        if not prev_image_inject:
            prev_image_inject = NoisedImageToInjectGroup()
        prev_image_inject = prev_image_inject.clone()
        to_inject = NoisedImageToInject(image=image, mask=mask_opt, vae=vae, invert_mask=invert_mask, resize_image=resize_image, strength_multival=strength_multival,
                                        start_percent=start_percent, guarantee_steps=guarantee_steps,
                                        img_inject_opts=img_inject_opts)
        prev_image_inject.add(to_inject)
        return (prev_image_inject,)

```
