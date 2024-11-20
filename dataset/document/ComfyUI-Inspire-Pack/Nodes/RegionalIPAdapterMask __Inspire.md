---
tags:
- IPAdapter
- RegionalImageProcessing
---

# Regional IPAdapter Mask (Inspire)
## Documentation
- Class name: `RegionalIPAdapterMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in applying regional IP adapter masks to influence the generation process, allowing for precise control over specific areas of an image during the generation or modification process. It leverages IP adapter conditioning techniques to integrate regional preferences or constraints, enhancing the customization and creativity of the output.
## Input types
### Required
- **`mask`**
    - The mask parameter specifies the area of the image to be influenced by the IP adapter, serving as a crucial factor in determining the scope and impact of the adaptation process.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`image`**
    - Image represents the target content that will be modified or generated, serving as the base for the application of the IP adapter mask.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`weight`**
    - Weight determines the intensity of the embedding's influence within the masked region, allowing for fine-tuning of the adaptation's effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - Noise adds a level of randomness to the adaptation process, offering a way to introduce variability and prevent overfitting to the embeddings.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Weight type specifies the method of applying weight to the embeddings, offering different strategies for integrating the embeddings into the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Start at indicates the initial point in the generation process at which the embeddings begin to influence the output, providing temporal control over the adaptation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - End at defines the final point in the generation process where the embeddings' influence ceases, allowing for precise temporal adaptation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`unfold_batch`**
    - Unfold batch is a boolean parameter that, when enabled, allows for batch processing of the adaptation, potentially improving efficiency. It affects the node's execution by enabling or disabling the processing of multiple inputs in a single batch, impacting the efficiency and speed of the adaptation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`faceid_v2`**
    - FaceID v2 is an optional boolean parameter that, when enabled, applies a second version of face identification technology to enhance the adaptation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`weight_v2`**
    - Weight v2 allows for an alternative weighting mechanism to be applied, offering additional flexibility in how the adaptation's effect is calibrated.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`combine_embeds`**
    - Combine embeds specifies the method for combining multiple embeddings, providing strategies such as concatenation or averaging to integrate different attributes or styles.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`neg_image`**
    - Neg image allows for the specification of a negative image, offering a way to explicitly discourage certain attributes or styles within the masked region.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`regional_ipadapter`**
    - Comfy dtype: `REGIONAL_IPADAPTER`
    - The output is a conditioned model adapted to the specified regional preferences or constraints, ready for further processing or generation tasks.
    - Python dtype: `IPAdapterConditioning`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalIPAdapterMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "image": ("IMAGE",),
                "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight_type": (["original", "linear", "channel penalty"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "unfold_batch": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "faceid_v2": ("BOOLEAN", {"default": False}),
                "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "neg_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("REGIONAL_IPADAPTER", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(mask, image, weight, noise, weight_type, start_at=0.0, end_at=1.0, unfold_batch=False, faceid_v2=False, weight_v2=False, combine_embeds="concat", neg_image=None):
        cond = IPAdapterConditioning(mask, weight, weight_type, noise=noise, image=image, neg_image=neg_image, start_at=start_at, end_at=end_at, unfold_batch=unfold_batch, weight_v2=weight_v2, combine_embeds=combine_embeds)
        return (cond, )

```
