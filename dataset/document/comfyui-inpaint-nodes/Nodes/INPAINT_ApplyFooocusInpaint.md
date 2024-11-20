---
tags:
- DepthMap
- Image
- Inpaint
---

# Apply Fooocus Inpaint
## Documentation
- Class name: `INPAINT_ApplyFooocusInpaint`
- Category: `inpaint`
- Output node: `False`

This node applies a specialized inpainting technique using the Fooocus method to enhance or modify images based on provided patches and latent information. It integrates seamlessly with the inpainting process, leveraging learned features and adjustments to achieve high-quality inpainting results.
## Input types
### Required
- **`model`**
    - The model parameter represents the base model to which the inpainting patches and adjustments will be applied. It is crucial for defining the starting point of the inpainting process.
    - Comfy dtype: `MODEL`
    - Python dtype: `ModelPatcher`
- **`patch`**
    - This parameter consists of the inpainting head model and the LoRA patch information, which are essential for applying the specific inpainting adjustments to the base model.
    - Comfy dtype: `INPAINT_PATCH`
    - Python dtype: `tuple[InpaintHead, dict[str, Tensor]]`
- **`latent`**
    - The latent parameter contains the latent representation and noise mask of the image to be inpainted, providing the necessary context for the inpainting operation.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict[str, Any]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns a modified version of the base model with inpainting patches applied, ready for further inpainting tasks.
    - Python dtype: `ModelPatcher`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ApplyFooocusInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "patch": ("INPAINT_PATCH",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    CATEGORY = "inpaint"
    FUNCTION = "patch"

    _inpaint_head_feature: Tensor | None = None
    _inpaint_block: Tensor | None = None

    def patch(
        self,
        model: ModelPatcher,
        patch: tuple[InpaintHead, dict[str, Tensor]],
        latent: dict[str, Any],
    ):
        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()

        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

        inpaint_head_model, inpaint_lora = patch
        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        self._inpaint_head_feature = inpaint_head_model(feed)
        self._inpaint_block = None

        lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        lora_keys.update({x: x for x in base_model.state_dict().keys()})
        loaded_lora = load_fooocus_patch(inpaint_lora, lora_keys)

        m = model.clone()
        m.set_model_input_block_patch(self._input_block_patch)
        patched = m.add_patches(loaded_lora, 1.0)

        not_patched_count = sum(1 for x in loaded_lora if x not in patched)
        if not_patched_count > 0:
            print(f"[ApplyFooocusInpaint] Failed to patch {not_patched_count} keys")

        inject_patched_calculate_weight()
        return (m,)

    def _input_block_patch(self, h: Tensor, transformer_options: dict):
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                assert self._inpaint_head_feature is not None
                batch = h.shape[0] // self._inpaint_head_feature.shape[0]
                self._inpaint_block = self._inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h

```
