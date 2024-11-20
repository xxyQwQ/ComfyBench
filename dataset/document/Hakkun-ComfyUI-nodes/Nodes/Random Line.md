# Random Line
## Documentation
- Class name: `Random Line`
- Category: `Hakkun`
- Output node: `False`

The Random Line node is designed to select a random line from a given text input based on a specified seed. This functionality allows for the introduction of randomness and variability in text processing tasks, ensuring that outputs can vary even with the same input under different seeds.
## Input types
### Required
- **`text`**
    - The 'text' parameter is the input text from which a random line will be selected. It supports multiline input, allowing for a broader range of text to be processed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - The 'seed' parameter determines the randomness of the line selection, ensuring reproducibility of results when the same seed is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Returns a randomly selected line from the input text, ensuring variability in the output based on the provided seed.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RandomLine:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (TEXT_TYPE, {"default": '', "multiline": True}),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "get_random_line"

    CATEGORY = "Hakkun"

    def get_random_line(self, text, seed):
        return get_random_line(text, seed),

```
