# Text Padder (Mikey)
## Documentation
- Class name: `TextPadderMikey`
- Category: `Mikey/Text`
- Output node: `False`

The TextPadderMikey node is designed to adjust the length of a given text to a specified target length using either padding or repeating techniques. It allows for customization of the padding character, offering flexibility in text formatting for various applications.
## Input types
### Required
- **`text`**
    - The input text to be padded or repeated to reach the desired length. This parameter is central to the node's operation, determining the base content that will be manipulated.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`length`**
    - Specifies the target length for the output text. This parameter directly influences the amount of padding or repetition applied to the input text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`technique`**
    - Determines the method used to extend the text to the desired length, either by padding with a character or repeating the text.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`padding_character`**
    - The character used for padding when the 'pad' technique is selected. This allows for customization of the padding process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`padded_text`**
    - Comfy dtype: `STRING`
    - The resulting text after applying the specified padding or repeating technique to reach the desired length.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TextPadderMikey:
    # Pad text to a fixed length with a specified padding character
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'default': 'text to pad'}),
                             'length': ('INT', {'default': 512, 'min': 1, 'max': 1000}),
                             'technique': (['pad','repeat'], {'default': 'pad'}),
                             'padding_character': ('STRING', {'default': ','})}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('padded_text',)
    FUNCTION = 'pad_text'
    CATEGORY = 'Mikey/Text'

    def pad_text(self, text, length, technique, padding_character):
        if len(text) >= length:
            return (text,)
        if technique == 'pad':
            padded_text = text.ljust(length, padding_character)
        elif technique == 'repeat':
            # repeat but don't cut off the text
            padded_text = text * (length // len(text))
            # fill the rest with the padding character
            padded_text = padded_text.ljust(length, padding_character)
        return (padded_text,)

```
