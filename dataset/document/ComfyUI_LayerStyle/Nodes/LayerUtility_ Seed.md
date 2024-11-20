# LayerUtility: Seed
## Documentation
- Class name: `LayerUtility: Seed`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The Seed node is designed to initialize or set a seed value for processes that require deterministic outcomes. It ensures that operations which involve randomness can be replicated exactly by providing a consistent starting point.
## Input types
### Required
- **`seed`**
    - The seed parameter is used to set the initial value for the seed, influencing the deterministic nature of subsequent operations. It ensures that results can be consistently replicated when the same seed is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`seed`**
    - Comfy dtype: `INT`
    - Outputs the same seed value that was input, allowing it to be used in subsequent operations that require a deterministic starting point.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SeedNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        return {"required": {
                "seed":("INT", {"default": 0, "min": 0, "max": 99999999999999999999, "step": 1}),
            },}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = 'seed_node'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def seed_node(self, seed):
        return (seed,)

```
