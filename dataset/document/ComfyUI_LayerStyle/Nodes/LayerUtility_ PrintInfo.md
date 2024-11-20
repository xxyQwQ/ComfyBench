# LayerUtility: PrintInfo
## Documentation
- Class name: `LayerUtility: PrintInfo`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `True`

The 'LayerUtility: PrintInfo' node is designed to log and return information about the input it receives, including the type of the input, dimensions and shape if it's a tensor, and serialization of the input if possible. It's a utility node aimed at providing insights into the data flowing through a node pipeline, facilitating debugging and data inspection.
## Input types
### Required
- **`anything`**
    - Accepts any type of input, aiming to provide detailed information about it, including its type, dimensions if applicable, and a serialized form. This flexibility allows for a wide range of data types to be inspected, making it a versatile tool for debugging and data analysis.
    - Comfy dtype: `*`
    - Python dtype: `AnyType`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - Outputs a string containing detailed information about the input, including its type, dimensions if applicable, and serialized form, aiding in debugging and data inspection.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PrintInfo:

  @classmethod
  def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
    return {
      "required": {
        "anything": (any, {}),
      },
    }

  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("text",)
  FUNCTION = "print_info"
  CATEGORY = '😺dzNodes/LayerUtility/Data'
  OUTPUT_NODE = True

  def print_info(self, anything=None):
    value = f'PrintInfo:\nInput type = {type(anything)}'
    if isinstance(anything, torch.Tensor):
      value += f"\n Input dim = {anything.dim()}, shape[0] = {anything.shape[0]} \n"
      for i in range(anything.shape[0]):
        t = anything[i]
        image = tensor2pil(t)
        value += f'\n index {i}: Image.size = {image.size}, Image.mode = {image.mode}, dim = {t.dim()}, '
        for j in range(t.dim()):
          value += f'shape[{j}] = {t.shape[j]}, '
        # value += f'\n {t} \n'
    elif anything is not None:
      try:
        value = value + json.dumps(anything) + "\n"
      except Exception:
        try:
          value = value + str(anything) + "\n"
        except Exception:
          value = 'source exists, but could not be serialized.'
    else:
      value = 'source does not exist.'

    log(value)

    return (value,)

```
