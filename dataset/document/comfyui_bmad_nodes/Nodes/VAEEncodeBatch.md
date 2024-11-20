---
tags:
- VAE
---

# VAEEncodeBatch
## Documentation
- Class name: `VAEEncodeBatch`
- Category: `Bmad`
- Output node: `False`

The VAEEncodeBatch node is designed to encode a batch of images into their latent representations using a Variational Autoencoder (VAE). It processes multiple images sequentially, leveraging a VAE model to transform each image into a latent space vector, and then concatenates these vectors to form a batch of latent representations.
## Input types
### Required
- **`inputs_len`**
    - Specifies the number of images to encode into their latent representations. It determines the size of the batch that will be processed by the node.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`vae`**
    - The Variational Autoencoder (VAE) model used for encoding images into their latent representations. It is a crucial component that defines the encoding mechanism.
    - Comfy dtype: `VAE`
    - Python dtype: `object`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation of the encoded images. It is a batch of vectors in the latent space, representing the input images.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VAEEncodeBatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "inputs_len": ("INT", {"default": 3, "min": 2, "max": 32, "step": 1}),
            "vae": ("VAE",)
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = base_category_path

    def encode(self, inputs_len, vae, **kwargs):
        vae_encoder = nodes.VAEEncode()

        def get_latent(input_name):
            pixels = kwargs[input_name]
            pixels = vae_encoder.vae_encode_crop_pixels(pixels)
            return vae.encode(pixels[:, :, :, :3])

        latent = get_latent("image_1")
        for r in range(1, inputs_len):
            latent = torch.cat([latent, get_latent(f"image_{r + 1}")], dim=0)

        return ({"samples": latent},)

```
