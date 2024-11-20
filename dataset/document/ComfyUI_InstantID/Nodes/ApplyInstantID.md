---
tags:
- IdentityImage
---

# Apply InstantID
## Documentation
- Class name: `ApplyInstantID`
- Category: `InstantID`
- Output node: `False`

The ApplyInstantID node is designed to integrate InstantID technology into images, leveraging facial analysis and control networks to enhance or modify the image based on specified conditions. It utilizes a combination of model inputs and image processing techniques to apply identity-preserving transformations, ensuring the output aligns with user-defined positive and negative conditioning.
## Input types
### Required
- **`instantid`**
    - Represents the InstantID model configuration and weights, crucial for the identity transformation process.
    - Comfy dtype: `INSTANTID`
    - Python dtype: `dict`
- **`insightface`**
    - Facial analysis model used to extract facial features from the image, essential for guiding the InstantID transformation.
    - Comfy dtype: `FACEANALYSIS`
    - Python dtype: `dict`
- **`control_net`**
    - Control network model that influences the strength and direction of the applied transformations.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `dict`
- **`image`**
    - The input image to be processed and transformed by the InstantID technology.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`model`**
    - The underlying model used for generating the transformations.
    - Comfy dtype: `MODEL`
    - Python dtype: `dict`
- **`positive`**
    - Positive conditioning phrases that guide the transformation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list of tuples`
- **`negative`**
    - Negative conditioning phrases that guide the transformation away from undesired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list of tuples`
- **`weight`**
    - Overall weight of the transformation, affecting the intensity of the applied changes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Defines the starting point of the transformation process, allowing for gradual application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Defines the ending point of the transformation process, ensuring the transformation is applied within a specific range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`image_kps`**
    - Optional keypoints image for more precise facial feature alignment.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Optional mask to limit the transformation to specific areas of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`MODEL`**
    - Comfy dtype: `MODEL`
    - The modified model after applying the InstantID transformations.
    - Python dtype: `dict`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Updated positive conditioning reflecting the applied transformations.
    - Python dtype: `list of tuples`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Updated negative conditioning reflecting the applied transformations.
    - Python dtype: `list of tuples`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ApplyInstantID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "instantid": ("INSTANTID", ),
                "insightface": ("FACEANALYSIS", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "weight": ("FLOAT", {"default": .8, "min": 0.0, "max": 5.0, "step": 0.01, }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, }),
            },
            "optional": {
                "image_kps": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("MODEL", "positive", "negative", )
    FUNCTION = "apply_instantid"
    CATEGORY = "InstantID"

    def apply_instantid(self, instantid, insightface, control_net, image, model, positive, negative, start_at, end_at, weight=.8, ip_weight=None, cn_strength=None, noise=0.35, image_kps=None, mask=None, combine_embeds='average'):
        self.dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        self.device = comfy.model_management.get_torch_device()

        ip_weight = weight if ip_weight is None else ip_weight
        cn_strength = weight if cn_strength is None else cn_strength

        output_cross_attention_dim = instantid["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280
        clip_extra_context_tokens = 16

        face_embed = extractFeatures(insightface, image)
        if face_embed is None:
            raise Exception('Reference Image: No face detected.')

        # if no keypoints image is provided, use the image itself (only the first one in the batch)
        face_kps = extractFeatures(insightface, image_kps if image_kps is not None else image[0].unsqueeze(0), extract_kps=True)

        if face_kps is None:
            face_kps = torch.zeros_like(image) if image_kps is None else image_kps
            print(f"\033[33mWARNING: No face detected in the keypoints image!\033[0m")

        clip_embed = face_embed
        # InstantID works better with averaged embeds (TODO: needs testing)
        if clip_embed.shape[0] > 1:
            if combine_embeds == 'average':
                clip_embed = torch.mean(clip_embed, dim=0).unsqueeze(0)
            elif combine_embeds == 'norm average':
                clip_embed = torch.mean(clip_embed / torch.norm(clip_embed, dim=0, keepdim=True), dim=0).unsqueeze(0)

        if noise > 0:
            seed = int(torch.sum(clip_embed).item()) % 1000000007
            torch.manual_seed(seed)
            clip_embed_zeroed = noise * torch.rand_like(clip_embed)
            #clip_embed_zeroed = add_noise(clip_embed, noise)
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

        work_model = model.clone()

        sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

        if mask is not None:
            mask = mask.to(self.device)

        patch_kwargs = {
            "ipadapter": self.instantid,
            "weight": ip_weight,
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

        # 2: do the ControlNet
        if mask is not None and len(mask.shape) < 3:
            mask = mask.unsqueeze(0)

        cnets = {}
        cond_uncond = []

        is_cond = True
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(face_kps.movedim(-1,1), cn_strength, (start_at, end_at))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                d['cross_attn_controlnet'] = image_prompt_embeds.to(comfy.model_management.intermediate_device()) if is_cond else uncond_image_prompt_embeds.to(comfy.model_management.intermediate_device())

                if mask is not None and is_cond:
                    d['mask'] = mask
                    d['set_area_to_bounds'] = False

                n = [t[0], d]
                c.append(n)
            cond_uncond.append(c)
            is_cond = False

        return(work_model, cond_uncond[0], cond_uncond[1], )

```
