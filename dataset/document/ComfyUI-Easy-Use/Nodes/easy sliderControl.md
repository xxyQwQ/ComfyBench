# Easy Slider Control
## Documentation
- Class name: `easy sliderControl`
- Category: `EasyUse/Util`
- Output node: `True`

The 'easy sliderControl' node provides a user-friendly interface for dynamically adjusting parameters within the ComfyUI environment. It simplifies the process of fine-tuning values through sliders, enhancing the user's ability to experiment with and control the behavior of various features or models in real-time.
## Input types
### Required
- **`mode`**
    - Specifies the operational mode of the node, affecting how parameters are adjusted and applied within the ComfyUI environment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`model_type`**
    - Determines the type of model to which the adjustments will be applied, influencing the node's execution and the outcome of parameter modifications.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`layer_weights`**
    - Comfy dtype: `STRING`
    - Outputs the adjusted layer weights as determined by the slider controls, reflecting the changes made to the model's parameters.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class sliderControl:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (['ipadapter layer weights'],),
                "model_type": (['sdxl', 'sd1'],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_weights",)

    FUNCTION = "control"
    OUTPUT_NODE = True

    CATEGORY = "EasyUse/Util"

    def control(self, mode, model_type, prompt=None, my_unique_id=None, extra_pnginfo=None):
        values = ''
        if my_unique_id in prompt:
            print(prompt[my_unique_id])
            if 'values' in prompt[my_unique_id]["inputs"]:
                values = prompt[my_unique_id]["inputs"]['values']

        return (values,)

```
