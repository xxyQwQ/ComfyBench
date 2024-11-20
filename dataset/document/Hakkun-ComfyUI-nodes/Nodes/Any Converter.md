---
tags:
- DataTypeConversion
- NumericConversion
---

# Any Converter
## Documentation
- Class name: `Any Converter`
- Category: `Hakkun`
- Output node: `False`

The Any Converter node is designed to facilitate type conversion and processing of various input types, including integers, floats, numbers, strings, and seeds. It primarily serves to normalize and convert input data into multiple formats, ensuring compatibility and utility across different processing scenarios. This node is particularly useful in scenarios where dynamic type handling and conversion are required, providing a versatile tool for data manipulation and preparation.
## Input types
### Required
### Optional
- **`int_`**
    - An optional integer input that, if provided, is directly used as the value for conversion. It plays a crucial role in determining the output when present, overriding other inputs except for 'str_' if it is valid.
    - Comfy dtype: `INT`
    - Python dtype: `Optional[int]`
- **`float_`**
    - An optional floating-point input that is used for conversion if 'int_' is not provided. It allows for decimal values to be included in the conversion process, adding flexibility in handling numerical data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Optional[float]`
- **`number_`**
    - A generic numerical input that is used for conversion if neither 'int_' nor 'float_' are provided. It offers a broader range of numerical input handling, accommodating various formats.
    - Comfy dtype: `NUMBER`
    - Python dtype: `Optional[float]`
- **`string_`**
    - An optional string input that, if valid, is converted into multiple formats including integer, float, and seed. This input significantly expands the node's versatility by allowing text-based numerical conversion.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
- **`seed_`**
    - An optional seed input provided as a dictionary, used for conversion if no other inputs are valid. It extracts the 'seed' value for conversion, demonstrating the node's capability to handle structured input.
    - Comfy dtype: `SEED`
    - Python dtype: `Optional[Dict[str, int]]`
- **`str_`**
    - An alternative string input that can be used for conversion, similar to 'string_'. If valid, it takes precedence over other inputs except for 'int_', showcasing the node's flexibility in input prioritization.
    - Comfy dtype: `STR`
    - Python dtype: `Optional[str]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - The converted integer value from the input.
    - Python dtype: `int`
- **`float`**
    - Comfy dtype: `FLOAT`
    - The converted floating-point value from the input, provided twice for compatibility.
    - Python dtype: `float`
- **`number`**
    - Comfy dtype: `NUMBER`
    - The converted numerical value from the input, identical to 'float' for consistency.
    - Python dtype: `float`
- **`string`**
    - Comfy dtype: `STRING`
    - The original or converted string representation of the input.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `SEED`
    - A dictionary containing the converted integer value under the key 'seed', demonstrating the node's ability to output structured data.
    - Python dtype: `Dict[str, int]`
- **`str`**
    - Comfy dtype: `STR`
    - The original or converted string input, ensuring textual data is preserved or transformed as needed.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnyConverter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "int_": ("INT", {"forceInput": True}),
                "float_": ("FLOAT", {"forceInput": True}),
                "number_": ("NUMBER", {"forceInput": True}),
                "string_": (TEXT_TYPE, {"forceInput": True}),
                "seed_": ("SEED", ),
                "str_": ("STR", ),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "NUMBER", TEXT_TYPE, "SEED", "STR")
    FUNCTION = "convert"
    CATEGORY = "Hakkun"

    def string_to_int(self, s):
        try:
            return int(float(s))
        except (ValueError, TypeError):
            return 0

    def string_to_number(self, s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return 0.0

    def convert(self, int_=None, float_=None, number_=None, string_=None, seed_=None, str_=None):
        if isOk(str_):
            string_=str_
        if isOk(int_):
            value=int_
        elif isOk(float_):
            value=float_
        elif isOk(number_):
            value=number_
        elif isOk(string_):
            return (self.string_to_int(string_),
                    self.string_to_number(string_),
                    self.string_to_number(string_),
                    string_,
                    {"seed":self.string_to_int(string_), },
                    string_,)
        elif isOk(seed_):
            value=seed_.get('seed')
        else:
            value=0
        return int(value),float(value),float(value),str(value),{"seed":int(value), }

```
