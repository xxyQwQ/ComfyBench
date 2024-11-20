---
tags:
- ModelGuidance
---

# Self-Attention Guidance
## Documentation
- Class name: `SelfAttentionGuidance`
- Category: `_for_testing`
- Output node: `False`

This node enhances the self-attention mechanism within neural networks by recording and potentially modifying attention scores based on specific conditions. It aims to improve model performance and interpretability by providing insights into how different parts of the input are weighted during the attention process.
## Input types
### Required
- **`model`**
    - The neural network model to which self-attention guidance will be applied. It is crucial for enabling the node to interact with and modify the model's attention mechanisms.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`scale`**
    - A scaling factor that adjusts the intensity of the attention modification. It plays a significant role in determining how much the original attention scores are altered.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`blur_sigma`**
    - Specifies the standard deviation of the Gaussian blur applied to attention scores. This parameter influences the smoothness of the attention distribution, affecting the model's focus on input features.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified neural network model with enhanced self-attention mechanisms. This output reflects the adjustments made to the model's attention scores, aiming to improve focus and interpretability.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - Reroute
    - [ADE_AnimateDiffLoaderWithContext](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_AnimateDiffLoaderWithContext.md)
    - Attention couple
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)



## Source code
```python
class SelfAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.1}),
                             "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale, blur_sigma):
        m = model.clone()

        attn_scores = None

        # TODO: make this work properly with chunked batches
        #       currently, we can only save the attn from one UNet call
        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            # if uncond, save the attention scores
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            b = q.shape[0] // len(cond_or_uncond)
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # do the entire attention operation, but save the attention scores to attn_scores
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])
                # when using a higher batch size, I BELIEVE the result batch dimension is [uc1, ... ucn, c1, ... cn]
                n_slices = heads * b
                attn_scores = sim[n_slices * uncond_index:n_slices * (uncond_index+1)]
                return out
            else:
                return optimized_attention(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])

        def post_cfg_function(args):
            nonlocal attn_scores
            uncond_attn = attn_scores

            sag_scale = scale
            sag_sigma = blur_sigma
            sag_threshold = 1.0
            model = args["model"]
            uncond_pred = args["uncond_denoised"]
            uncond = args["uncond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]
            if min(cfg_result.shape[2:]) <= 4: #skip when too small to add padding
                return cfg_result

            # create the adversarially blurred image
            degraded = create_blur_map(uncond_pred, uncond_attn, sag_sigma, sag_threshold)
            degraded_noised = degraded + x - uncond_pred
            # call into the UNet
            (sag,) = comfy.samplers.calc_cond_batch(model, [uncond], degraded_noised, sigma, model_options)
            return cfg_result + (degraded - sag) * sag_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)

        # from diffusers:
        # unet.mid_block.attentions[0].transformer_blocks[0].attn1.patch
        m.set_model_attn1_replace(attn_and_record, "middle", 0, 0)

        return (m, )

```
