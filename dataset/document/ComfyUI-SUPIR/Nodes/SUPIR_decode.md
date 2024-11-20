---
tags:
- VAE
---

# SUPIR Decode
## Documentation
- Class name: `SUPIR_decode`
- Category: `SUPIR`
- Output node: `False`

The SUPIR_decode node is designed for decoding latent representations into images using a specific VAE model within the SUPIR framework. It focuses on reconstructing images from their encoded state, potentially after modifications or processing in the latent space, thereby enabling the generation of images or the restoration of details from compressed or abstract representations.
## Input types
### Required
- **`SUPIR_VAE`**
    - The VAE model used for decoding. It is crucial for the decoding process as it defines the architecture and parameters that will be used to reconstruct the image from its latent representation.
    - Comfy dtype: `SUPIRVAE`
    - Python dtype: `torch.nn.Module`
- **`latents`**
    - The latent representation of an image that is to be decoded. This input is essential for the reconstruction process, as it contains the compressed information that will be expanded back into an image.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`use_tiled_vae`**
    - A flag indicating whether to use a tiled approach for the VAE decoding. This affects the decoding process by potentially improving the handling of large images or enhancing performance on certain hardware configurations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`decoder_tile_size`**
    - Specifies the size of the tiles when using a tiled VAE decoding approach. This parameter is critical for determining the resolution and detail in the reconstructed image, affecting the overall quality and efficiency of the decoding process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image resulting from the decoding process. It represents the final output where the latent representation has been transformed back into visual form, showcasing the reconstructed or generated image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SUPIR_decode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_VAE": ("SUPIRVAE",),
            "latents": ("LATENT",),
            "use_tiled_vae": ("BOOLEAN", {"default": True}),
            "decoder_tile_size": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "SUPIR"

    def decode(self, SUPIR_VAE, latents, use_tiled_vae, decoder_tile_size):
        device = mm.get_torch_device()
        mm.unload_all_models()
        samples = latents["samples"]
        
        B, H, W, C = samples.shape
                
        pbar = comfy.utils.ProgressBar(B)
       
        if mm.should_use_bf16():
            print("Decoder using bf16")
            dtype = torch.bfloat16
        else:
            print("Decoder using fp32")
            dtype = torch.float32
        print("SUPIR decoder using", dtype)
           
        SUPIR_VAE.to(dtype).to(device)
        samples = samples.to(device)

        if use_tiled_vae:
            from .SUPIR.utils.tilevae import VAEHook
            # Store the `original_forward` only if it hasn't been stored already
            if not hasattr(SUPIR_VAE.decoder, 'original_forward'):
                SUPIR_VAE.decoder.original_forward = SUPIR_VAE.decoder.forward
            SUPIR_VAE.decoder.forward = VAEHook(
                SUPIR_VAE.decoder, decoder_tile_size // 8, is_decoder=True, fast_decoder=False,
                fast_encoder=False, color_fix=False, to_gpu=True)
        else:
            # Only assign `original_forward` back if it exists
            if hasattr(SUPIR_VAE.decoder, 'original_forward'):
                SUPIR_VAE.decoder.forward = SUPIR_VAE.decoder.original_forward

        out = []
        for sample in samples:
            autocast_condition = (dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
            with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
                sample = 1.0 / 0.13025 * sample
                decoded_image = SUPIR_VAE.decode(sample.unsqueeze(0))
                out.append(decoded_image)
                pbar.update(1)

        decoded_out= torch.cat(out, dim=0).float()

        if "original_size" in latents and latents["original_size"] is not None:
            orig_H, orig_W = latents["original_size"]
            if decoded_out.shape[2] != orig_H or decoded_out.shape[3] != orig_W:
                print("Restoring original dimensions: ", orig_W,"x",orig_H)
                decoded_out = F.interpolate(decoded_out, size=(orig_H, orig_W), mode="bicubic")

        decoded_out = torch.clip(decoded_out, 0, 1)
        decoded_out = decoded_out.cpu().to(torch.float32).permute(0, 2, 3, 1)
        

        return (decoded_out,)

```
