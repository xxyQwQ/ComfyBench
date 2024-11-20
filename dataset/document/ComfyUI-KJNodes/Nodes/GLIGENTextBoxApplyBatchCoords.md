---
tags:
- CLIP
- Conditioning
---

# GLIGENTextBoxApplyBatchCoords
## Documentation
- Class name: `GLIGENTextBoxApplyBatchCoords`
- Category: `KJNodes/experimental`
- Output node: `False`

The GLIGENTextBoxApplyBatchCoords node is designed for scheduling the positions of GLIGEN text boxes within a batch, specifically tailored for use with AnimateDiff-Evolved. It facilitates the integration of text box positioning with the Spline Editor node, enabling dynamic text placement in generated images. This node processes batched coordinates and text inputs to compute positioning parameters, enhancing the visual storytelling capabilities of GLIGEN-generated content.
## Input types
### Required
- **`conditioning_to`**
    - The target for applying the conditioning, typically a sequence of conditioning layers or parameters.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Any]`
- **`latents`**
    - Used to determine the batch size for processing, indicating the number of latent representations to be positioned.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`clip`**
    - A standard text encoder used for encoding the text inputs, ensuring compatibility with the main prompt processing.
    - Comfy dtype: `CLIP`
    - Python dtype: `transformers.PreTrainedModel`
- **`gligen_textbox_model`**
    - Connects to the GLIGEN Loader, enabling the application of GLIGEN-specific text box positioning logic.
    - Comfy dtype: `GLIGEN`
    - Python dtype: `Any`
- **`coordinates`**
    - Accepts a JSON string of points for text box positioning, directly compatible with the Spline Editor node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text`**
    - Specifies the text content for which positions are being set within the GLIGEN bounding box.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - Defines the width of the GLIGEN bounding box for text placement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the GLIGEN bounding box for text placement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`size_multiplier`**
    - A factor that scales the size of the GLIGEN bounding box, affecting the overall dimensions of the text placement area.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Provides the conditioning parameters for the clip text encoder and sampler, facilitating the integration of positioned text into the generation process.
    - Python dtype: `List[Any]`
- **`coord_preview`**
    - Comfy dtype: `IMAGE`
    - An optional output offering a visual preview of the text box coordinates and bounding boxes.
    - Python dtype: `Any`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GLIGENTextBoxApplyBatchCoords:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING", ),
                              "latents": ("LATENT", ),
                              "clip": ("CLIP", ),
                              "gligen_textbox_model": ("GLIGEN", ),
                              "coordinates": ("STRING", {"forceInput": True}),
                              "text": ("STRING", {"multiline": True}),
                              "width": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                              "height": ("INT", {"default": 128, "min": 8, "max": 4096, "step": 8}),
                            },
                "optional": {"size_multiplier": ("FLOAT", {"default": [1.0], "forceInput": True})},
                }
    RETURN_TYPES = ("CONDITIONING", "IMAGE", )
    RETURN_NAMES = ("conditioning", "coord_preview", )
    FUNCTION = "append"
    CATEGORY = "KJNodes/experimental"
    DESCRIPTION = """
This node allows scheduling GLIGEN text box positions in a batch,  
to be used with AnimateDiff-Evolved. Intended to pair with the  
Spline Editor -node.  

GLIGEN model can be downloaded through the Manage's "Install Models" menu.  
Or directly from here:  
https://huggingface.co/comfyanonymous/GLIGEN_pruned_safetensors/tree/main  
  
Inputs:  
- **latents** input is used to calculate batch size  
- **clip** is your standard text encoder, use same as for the main prompt  
- **gligen_textbox_model** connects to GLIGEN Loader  
- **coordinates** takes a json string of points, directly compatible  
with the spline editor node.
- **text** is the part of the prompt to set position for  
- **width** and **height** are the size of the GLIGEN bounding box  
  
Outputs:
- **conditioning** goes between to clip text encode and the sampler  
- **coord_preview** is an optional preview of the coordinates and  
bounding boxes.

"""

    def append(self, latents, coordinates, conditioning_to, clip, gligen_textbox_model, text, width, height, size_multiplier=[1.0]):
        coordinates = json.loads(coordinates.replace("'", '"'))
        coordinates = [(coord['x'], coord['y']) for coord in coordinates]

        batch_size = sum(tensor.size(0) for tensor in latents.values())
        if len(coordinates) != batch_size:
            print("GLIGENTextBoxApplyBatchCoords WARNING: The number of coordinates does not match the number of latents")

        c = []
        _, cond_pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)

        for t in conditioning_to:
            n = [t[0], t[1].copy()]
            
            position_params_batch = [[] for _ in range(batch_size)]  # Initialize a list of empty lists for each batch item
            if len(size_multiplier) != batch_size:
                size_multiplier = size_multiplier * (batch_size // len(size_multiplier)) + size_multiplier[:batch_size % len(size_multiplier)]

            for i in range(batch_size):
                x_position, y_position = coordinates[i]
                position_param = (cond_pooled, int((height // 8) * size_multiplier[i]), int((width // 8) * size_multiplier[i]), (y_position - height // 2) // 8, (x_position - width // 2) // 8)
                position_params_batch[i].append(position_param)  # Append position_param to the correct sublist

            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]
            else:
                prev = [[] for _ in range(batch_size)]
            # Concatenate prev and position_params_batch, ensuring both are lists of lists
            # and each sublist corresponds to a batch item
            combined_position_params = [prev_item + batch_item for prev_item, batch_item in zip(prev, position_params_batch)]
            n[1]['gligen'] = ("position_batched", gligen_textbox_model, combined_position_params)
            c.append(n)

        image_height = latents['samples'].shape[-2] * 8
        image_width = latents['samples'].shape[-1] * 8
        plot_image_tensor = plot_coordinates_to_tensor(coordinates, image_height, image_width, height, width, size_multiplier, text)
        
        return (c, plot_image_tensor,)

```
