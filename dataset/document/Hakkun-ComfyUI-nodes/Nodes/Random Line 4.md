# Random Line 4
## Documentation
- Class name: `Random Line 4`
- Category: `Hakkun`
- Output node: `False`

The Random Line 4 node is designed to select a random line from each of up to four provided text inputs based on a given seed, concatenate these lines with a specified delimiter, and return the concatenated string. This node enables the generation of unique and varied text outputs by randomly sampling from the provided inputs.
## Input types
### Required
- **`text1`**
    - A multiline text input from which a random line will be selected. It contributes to the node's output by potentially being one of the sources for a random line.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text2`**
    - A multiline text input, serving as another potential source for a random line selection.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text3`**
    - Serves as a potential source for the random line selection, similar to the other text inputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text4`**
    - Another multiline text input that can contribute a random line to the node's output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - The seed for the random number generator, ensuring reproducibility of the output across runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`delimiter`**
    - The delimiter used to concatenate the selected random lines from the text inputs.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The concatenated string formed by joining selected random lines from the provided text inputs with the specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RandomLine4:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": (TEXT_TYPE, {"default": '', "multiline": True}),
                "text2": (TEXT_TYPE, {"default": '', "multiline": True}),
                "text3": (TEXT_TYPE, {"default": '', "multiline": True}),
                "text4": (TEXT_TYPE, {"default": '', "multiline": True}),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "delimiter": (TEXT_TYPE, {"default": ', ', "multiline": False}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "get_random_line4"

    CATEGORY = "Hakkun"

    def get_random_line(self, textt, seed):
        lines = textt.splitlines()
        random.seed(seed)
        random_line = random.choice(lines)
        return random_line

    def get_random_line4(self, seed, delimiter, text1='', text2='', text3='', text4=''):
        random.seed(seed)

        texts = []
        if isOk(text1): texts.append(get_random_line(text1,seed))
        if isOk(text2): texts.append(get_random_line(text2,seed))
        if isOk(text3): texts.append(get_random_line(text3,seed))
        if isOk(text4): texts.append(get_random_line(text4,seed))

        delimiter = delimiter.replace("\\n", "\n")

        text = delimiter.join(texts)

        return text,

```
