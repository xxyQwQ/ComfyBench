---
tags:
- IPAdapter
---

# IPAdapter Advanced
## Documentation
- Class name: `IPAdapterAdvanced`
- Category: `ipadapter`
- Output node: `False`

The IPAdapterAdvanced node is designed to enhance image processing capabilities by applying advanced image processing techniques. It extends the functionality of simpler IP adapters, offering more sophisticated options for image manipulation and enhancement.
## Input types
### Required
- **`model`**
    - The model parameter specifies the neural network model to be used for image processing. It plays a crucial role in determining the quality and type of image enhancements applied.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`ipadapter`**
    - This parameter represents the specific IP adapter to be applied, dictating the nature of the image processing technique used.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `IPAdapter`
- **`image`**
    - The input image to be processed, serving as the base for all applied enhancements.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`weight`**
    - Controls the intensity of the applied image processing effect, offering flexibility in the final image output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the method used to calculate the weight of the image processing effect, affecting how the effect is applied over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`combine_embeds`**
    - Determines how multiple embeddings are combined during the image processing, impacting the final image's characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Defines the starting point of the effect applied by the IP adapter, allowing for fine-tuned control over the image processing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Specifies the endpoint of the effect applied, enabling precise manipulation of the image enhancement process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Specifies the scaling method for embeddings, affecting the influence of different components in the image processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image_negative`**
    - An optional image used to specify negative effects, enabling the exclusion of certain aspects from the final output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`attn_mask`**
    - An optional attention mask for more precise control over the areas affected by the image processing.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`clip_vision`**
    - Optional parameter for integrating CLIP vision models, enhancing the semantic understanding of the image.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `torch.Tensor`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the modified model after applying the IP adapter, reflecting the enhancements made to the image processing capabilities.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterAdvanced:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None):
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        if image_style is not None: # we are doing style + composition transfer
            if not is_sdxl:
                raise Exception("Style + Composition transfer is only available for SDXL models at the moment.") # TODO: check feasibility for SD1.5 models

            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style

            weight_type = "strong style and composition" if expand_style else "style and composition"
        if ipadapter_params is not None: # we are doing batch processing
            image = ipadapter_params['image']
            attn_mask = ipadapter_params['attn_mask']
            weight = ipadapter_params['weight']
            weight_type = ipadapter_params['weight_type']
            start_at = ipadapter_params['start_at']
            end_at = ipadapter_params['end_at']
        else:
            # at this point weight can be a list from the batch-weight or a single float
            weight = [weight]

        image = image if isinstance(image, list) else [image]

        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter_model, clip_vision, **ipa_args)

        del ipadapter
        return (work_model, face_image, )

```
