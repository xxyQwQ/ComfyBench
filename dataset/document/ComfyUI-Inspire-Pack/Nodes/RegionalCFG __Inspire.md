# Regional CFG (Inspire)
## Documentation
- Class name: `RegionalCFG __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

The RegionalCFG node in the Inspire Pack is designed to apply conditional generative filtering based on regional inputs. It modifies a given model's behavior to incorporate regional conditioning, allowing for localized adjustments in the generation process. This is achieved by applying a mask to conditionally influence the model's output, enabling fine-grained control over specific regions of interest.
## Input types
### Required
- **`model`**
    - The model parameter represents the generative model to be modified with regional conditioning. It is crucial for defining the base generative process that will be adjusted according to the regional mask.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`mask`**
    - The mask parameter is used to specify the regions of interest within the generative process. It plays a key role in localizing the effects of conditioning, allowing for targeted modifications in the model's output.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The output is a modified version of the input model, now equipped with the ability to perform regional conditioning. This enables localized adjustments in the generation process, based on the provided mask.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "mask": ("MASK",),
                             }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(model, mask):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        size = None

        def regional_cfg(args):
            nonlocal mask
            nonlocal size

            x = args['input']

            if mask.device != x.device:
                mask = mask.to(x.device)

            if size != (x.shape[2], x.shape[3]):
                size = (x.shape[2], x.shape[3])
                mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]

            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * mask

            return x - cfg_result

        m = model.clone()
        m.set_model_sampler_cfg_function(regional_cfg)
        return (m,)

```
