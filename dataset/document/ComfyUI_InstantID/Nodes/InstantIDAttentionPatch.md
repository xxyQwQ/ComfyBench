---
tags:
- ModelGuidance
- ModelPatch
---

# InstantID Patch Attention
## Documentation
- Class name: `InstantIDAttentionPatch`
- Category: `InstantID`
- Output node: `False`

The InstantIDAttentionPatch node is designed to enhance the attention mechanism within a model by applying specialized patches. These patches adjust the model's attention based on various factors such as image embeddings, noise levels, and specific conditions, aiming to improve the model's focus and performance on tasks related to image and face recognition.
## Input types
### Required
- **`instantid`**
    - The instantid parameter represents the core configuration or model that the attention patching will be applied to, serving as the foundation for the modifications.
    - Comfy dtype: `INSTANTID`
    - Python dtype: `dict`
- **`insightface`**
    - The insightface parameter is used for face analysis and feature extraction, providing critical data that influences how the attention patches are applied.
    - Comfy dtype: `FACEANALYSIS`
    - Python dtype: `module`
- **`image`**
    - The image parameter is the input image on which face analysis and feature extraction are performed, directly impacting the attention patching process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`model`**
    - The model parameter refers to the neural network model that will be modified by the attention patches, affecting its attention mechanism.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`weight`**
    - The weight parameter controls the influence of the attention patches on the model, allowing for fine-tuning of the patching effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - The start_at parameter specifies the beginning point in the model's processing where the attention patches start to take effect, marking the initial phase of modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - The end_at parameter defines the endpoint in the model's processing where the attention patches conclude their effect, delineating the final phase of modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - The noise parameter introduces randomness into the attention patching process, potentially enhancing the model's robustness and ability to generalize.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask`**
    - The mask parameter allows for selective application of the attention patches, targeting specific areas or features within the image for focused modification.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - This output type represents the modified model after the application of attention patches, reflecting the adjustments made to its attention mechanism.
    - Python dtype: `torch.nn.Module`
- **`face_embeds`**
    - Comfy dtype: `FACE_EMBEDS`
    - This output type represents the face embeddings generated or utilized during the patching process, which are critical for the targeted modifications.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InstantIDAttentionPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID", ),
                "insightface": ("FACEANALYSIS", ),
                "image": ("IMAGE", ),
                "model": ("MODEL", ),
                "weight": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 3.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "FACE_EMBEDS")
    FUNCTION = "patch_attention"
    CATEGORY = "InstantID"

    def patch_attention(self, instantid, insightface, image, model, weight, start_at, end_at, noise=0.0, mask=None):
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        self.device = comfy.model_management.get_torch_device()

        output_cross_attention_dim = instantid["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280
        clip_extra_context_tokens = 16

        face_embed = extractFeatures(insightface, image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
        else:
            clip_embed_zeroed = torch.zeros_like(clip_embed)

        clip_embeddings_dim = face_embed.shape[-1]

        # 1: patch the attention
        self.instantid = InstantID(
            instantid,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
        )

        self.instantid.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.instantid.get_image_embeds(clip_embed.to(self.device, dtype=self.dtype), clip_embed_zeroed.to(self.device, dtype=self.dtype))

        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        if weight == 0:
            return (model, { "cond": image_prompt_embeds, "uncond": uncond_image_prompt_embeds } )

        work_model = model.clone()

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(self.device)

        patch_kwargs = {
            "weight": weight,
            "ipadapter": self.instantid,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "mask": mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
        }

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                _set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            _set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
            number += 1

        return(work_model, { "cond": image_prompt_embeds, "uncond": uncond_image_prompt_embeds }, )

```
