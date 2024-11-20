---
tags:
- DepthMap
- Image
- ImagePreprocessing
---

# RGB SparseCtrl 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_SparseCtrlRGBPreprocessor`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess`
- Output node: `False`

The ACN_SparseCtrlRGBPreprocessor node is designed for preprocessing images for use with Sparse ControlNet models. It transforms input images into a format suitable for these models, specifically by encoding the images into a latent space that mimics the appearance of an image. This preprocessing step is crucial for the subsequent application of Sparse ControlNet techniques, ensuring compatibility and optimizing performance.
## Input types
### Required
- **`image`**
    - The input image to be preprocessed. This image is transformed to match the latent size requirements of the Sparse ControlNet model.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The Variational Autoencoder (VAE) model used for encoding the input image into a latent representation.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`latent_size`**
    - Specifies the size of the latent representation to which the input image should be resized and encoded.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
### Optional
- **`autosize`**
    - An optional parameter that allows for automatic resizing of the input image, with an option to specify padding.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `Dict[str, int]`
## Output types
- **`proc_IMAGE`**
    - Comfy dtype: `IMAGE`
    - The processed image, now encoded in a latent space that simulates the appearance of an image, ready for use with Sparse ControlNet models.
    - Python dtype: `PreprocSparseRGBWrapper`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RgbSparseCtrlPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "vae": ("VAE", ),
                "latent_size": ("LATENT", ),
            },
            "optional": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("proc_IMAGE",)
    FUNCTION = "preprocess_images"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl/preprocess"

    def preprocess_images(self, vae: VAE, image: Tensor, latent_size: Tensor):
        # first, resize image to match latents
        image = image.movedim(-1,1)
        image = comfy.utils.common_upscale(image, latent_size["samples"].shape[3] * 8, latent_size["samples"].shape[2] * 8, 'nearest-exact', "center")
        image = image.movedim(1,-1)
        # then, vae encode
        try:
            image = vae.vae_encode_crop_pixels(image)
        except Exception:
            image = VAEEncode.vae_encode_crop_pixels(image)
        encoded = vae.encode(image[:,:,:,:3])
        return (PreprocSparseRGBWrapper(condhint=encoded),)

```
