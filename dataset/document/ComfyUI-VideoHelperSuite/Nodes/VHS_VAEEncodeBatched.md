---
tags:
- VAE
---

# VAE Encode Batched 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_VAEEncodeBatched`
- Category: `Video Helper Suite 🎥🅥🅗🅢/batched nodes`
- Output node: `False`

This node is designed for batch processing of images through a Variational Autoencoder (VAE) to encode them into a latent space representation. It efficiently handles large sets of images by dividing them into smaller batches, encoding each batch separately, and then aggregating the results. This approach optimizes resource utilization and accelerates the encoding process, making it suitable for applications requiring the transformation of images into their latent representations for further processing or analysis.
## Input types
### Required
- **`pixels`**
    - The 'pixels' parameter represents the images to be encoded into latent space. It is crucial for defining the input data that will undergo the encoding process, directly influencing the node's output by determining the characteristics of the generated latent representations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The 'vae' parameter specifies the Variational Autoencoder model to be used for encoding the images. It plays a pivotal role in the transformation process, as the model's architecture and trained parameters directly affect the quality and characteristics of the encoded latent space.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`per_batch`**
    - The 'per_batch' parameter determines the number of images to be processed in each batch. It allows for flexible control over the batch size, balancing between computational efficiency and resource consumption.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a latent space representation of the input images, encoded by the specified VAE model. This representation is crucial for downstream tasks that require a compressed yet informative version of the original data.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class VAEEncodeBatched:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE", ), "vae": ("VAE", ),
                "per_batch": ("INT", {"default": 16, "min": 1})
                }
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/batched nodes"

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"

    def encode(self, vae, pixels, per_batch):
        t = []
        pbar = ProgressBar(pixels.shape[0])
        for start_idx in range(0, pixels.shape[0], per_batch):
            try:
                sub_pixels = vae.vae_encode_crop_pixels(pixels[start_idx:start_idx+per_batch])
            except:
                sub_pixels = VAEEncode.vae_encode_crop_pixels(pixels[start_idx:start_idx+per_batch])
            t.append(vae.encode(sub_pixels[:,:,:,:3]))
            pbar.update(per_batch)
        return ({"samples": torch.cat(t, dim=0)}, )

```
