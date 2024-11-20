---
tags:
- Conditioning
---

# SUPIR Conditioner
## Documentation
- Class name: `SUPIR_conditioner`
- Category: `SUPIR`
- Output node: `False`

The SUPIR_conditioner node is designed to condition the generation process in the SUPIR model, utilizing positive and negative prompts along with optional captions to influence the generated output. It plays a crucial role in refining and directing the model's output towards the desired thematic or stylistic outcomes.
## Input types
### Required
- **`SUPIR_model`**
    - The SUPIR model instance to be conditioned. It is central to the operation as it provides the necessary context and parameters for conditioning.
    - Comfy dtype: `SUPIRMODEL`
    - Python dtype: `object`
- **`latents`**
    - Latent representations that are to be conditioned. These play a key role in determining the final output by providing a base that is modified according to the prompts and captions.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`positive_prompt`**
    - A positive prompt guiding the model towards desired features or themes in the output. It significantly influences the conditioning process by highlighting what to include.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_prompt`**
    - A negative prompt indicating what features or themes to avoid in the model's output. It helps in steering the generation away from undesired elements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`captions`**
    - Optional captions that can be added to the positive prompt for more specific conditioning. When provided, they enhance the contextual understanding of the desired output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive`**
    - Comfy dtype: `SUPIR_cond_pos`
    - Conditioned latent representations with the influence of positive prompts. These are ready for further processing or generation steps.
    - Python dtype: `dict`
- **`negative`**
    - Comfy dtype: `SUPIR_cond_neg`
    - Conditioned latent representations influenced by negative prompts, serving as a contrast to the positive conditioning.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SUPIR_conditioner:
    # @classmethod
    # def IS_CHANGED(s):
    #     return ""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "SUPIR_model": ("SUPIRMODEL",),
            "latents": ("LATENT",),
            "positive_prompt": ("STRING", {"multiline": True, "default": "high quality, detailed", }),
            "negative_prompt": ("STRING", {"multiline": True, "default": "bad quality, blurry, messy", }),
        },
            "optional": {
                "captions": ("STRING", {"forceInput": True, "multiline": False, "default": "", }),
            }
        }

    RETURN_TYPES = ("SUPIR_cond_pos", "SUPIR_cond_neg",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "condition"
    CATEGORY = "SUPIR"
    DESCRIPTION = """
Creates the conditioning for the sampler.  
Caption input is optional, when it receives a single caption, it's added to the positive prompt.
    
If a list of caption is given for single input image, the captions need to match the number of tiles,  
refer to the SUPIR Tiles node.  
  
If a list of captions is given and it matches the incoming image batch, each image uses corresponding caption.
"""

    def condition(self, SUPIR_model, latents, positive_prompt, negative_prompt, captions=""):
        
        device = mm.get_torch_device()
        mm.soft_empty_cache()

        if "original_size" in latents:
            original_size = latents["original_size"]
            samples = latents["samples"]
        else:
            original_size = None
            samples = latents["samples"] * 0.13025
        
        N, H, W, C = samples.shape
        import copy

        if not isinstance(captions, list):
            captions_list = []
            captions_list.append([captions])
            captions_list = captions_list * N
        else:
            captions_list = captions

        print("captions: ", captions_list)
      
        SUPIR_model.conditioner.to(device)
        samples = samples.to(device)

        uc = []
        pbar = comfy.utils.ProgressBar(N)
        autocast_condition = (SUPIR_model.model.dtype != torch.float32) and not comfy.model_management.is_device_mps(device)
        with torch.autocast(comfy.model_management.get_autocast_device(device), dtype=SUPIR_model.model.dtype) if autocast_condition else nullcontext():
            if N != len(captions_list): #Tiled captioning
                print("Tiled captioning")
                c = []
                uc = []
                for i, caption in enumerate(captions_list):
                    cond = {}
                    cond['original_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['crop_coords_top_left'] = torch.tensor([[0, 0]]).to(device)
                    cond['target_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['aesthetic_score'] = torch.tensor([[9.0]]).to(device)
                    cond['control'] = samples[0].unsqueeze(0)

                    uncond = copy.deepcopy(cond)
                    uncond['txt'] = [negative_prompt]
                    
                    cond['txt'] = [''.join([caption[0], positive_prompt])]
                    if i == 0:
                        _c, uc = SUPIR_model.conditioner.get_unconditional_conditioning(cond, uncond)
                    else:
                        _c, _ = SUPIR_model.conditioner.get_unconditional_conditioning(cond, None)
    
                    c.append(_c)
                    pbar.update(1)
            else: #batch captioning
                print("Batch captioning")
                c = []
                uc = []
                for i, sample in enumerate(samples):
                    
                    cond = {}
                    cond['original_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['crop_coords_top_left'] = torch.tensor([[0, 0]]).to(device)
                    cond['target_size_as_tuple'] = torch.tensor([[1024, 1024]]).to(device)
                    cond['aesthetic_score'] = torch.tensor([[9.0]]).to(device)
                    cond['control'] = sample.unsqueeze(0)

                    uncond = copy.deepcopy(cond)
                    uncond['txt'] = [negative_prompt]
                    cond['txt'] = [''.join([captions_list[i][0], positive_prompt])]
                    _c, _uc = SUPIR_model.conditioner.get_unconditional_conditioning(cond, uncond)    
                    c.append(_c)
                    uc.append(_uc)
                    
                    pbar.update(1)

            
        SUPIR_model.conditioner.to('cpu')

        if "original_size" in latents:
            original_size = latents["original_size"]
        else:
            original_size = None
                
        return ({"cond": c, "original_size":original_size}, {"uncond": uc},)

```
