---
tags:
- ModelMerge
---

# ModelMergeSD3_2B
## Documentation
- Class name: `ModelMergeSD3_2B`
- Category: `advanced/model_merging/model_specific`
- Output node: `False`

This node specializes in merging two models by adjusting their embedding and block parameters specifically for the SD3_2B architecture. It allows for fine-tuned control over the merging process by providing adjustable parameters for positional, context, and various embedder components, as well as for each of the joint blocks and the final layer, enabling a highly customized integration of model features.
## Input types
### Required
- **`model1`**
    - The first model to be merged. It serves as the base model for the merging process.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.ModelBase`
- **`model2`**
    - The second model to be merged. Its features will be integrated into the first model based on the specified ratios.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.ModelBase`
- **`pos_embed.`**
    - Adjusts the merging ratio for the positional embedding parameters, influencing how positional information is integrated from both models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`x_embedder.`**
    - Controls the merging ratio for the x embedder, affecting the integration of x-axis positional information.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`context_embedder.`**
    - Determines the merging ratio for the context embedder, impacting how contextual information is blended between the two models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`y_embedder.`**
    - Sets the merging ratio for the y embedder, influencing the integration of y-axis positional information.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`t_embedder.`**
    - Adjusts the merging ratio for the time embedder, affecting how temporal information is merged.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`joint_blocks.i.`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`final_layer.`**
    - Controls the merging ratio for the final layer, determining how the final output features are combined.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The resulting model after merging, incorporating adjustments from both input models according to the specified ratios.
    - Python dtype: `comfy.model_base.ModelBase`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ModelMergeSD3_2B(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    CATEGORY = "advanced/model_merging/model_specific"

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["pos_embed."] = argument
        arg_dict["x_embedder."] = argument
        arg_dict["context_embedder."] = argument
        arg_dict["y_embedder."] = argument
        arg_dict["t_embedder."] = argument

        for i in range(24):
            arg_dict["joint_blocks.{}.".format(i)] = argument

        arg_dict["final_layer."] = argument

        return {"required": arg_dict}

```
