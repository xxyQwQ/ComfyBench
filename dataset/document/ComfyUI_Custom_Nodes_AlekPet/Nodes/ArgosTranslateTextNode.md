# Argos Translate Text Node
## Documentation
- Class name: `ArgosTranslateTextNode`
- Category: `AlekPet Nodes/text`
- Output node: `False`

This node specializes in translating text from one language to another, leveraging the Argos Translate library. It modifies the input types inherited from its parent class to exclude the 'clip' parameter, focusing solely on text translation without the need for additional conditioning inputs.
## Input types
### Required
- **`from_translate`**
    - Specifies the source language code from which the text will be translated. It plays a crucial role in determining the translation process by identifying the original language of the input text.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`to_translate`**
    - Defines the target language code to which the text will be translated. This parameter is essential for guiding the translation engine towards the desired output language.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text`**
    - The text to be translated. This parameter is central to the node's functionality, as it provides the content that will undergo the translation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The translated text. This output is the direct result of the translation process, providing users with text that has been converted from the source to the target language.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ArgosTranslateTextNode(ArgosTranslateCLIPTextEncodeNode):
    @classmethod
    def INPUT_TYPES(self):
        return_types = super().INPUT_TYPES()
        del return_types["required"]["clip"]
        return return_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "argos_translate_text"

    CATEGORY = "AlekPet Nodes/text"

    def argos_translate_text(self, from_translate, to_translate, text):
        self.langs_support = ALL_CODES[from_translate]["targets"]
        text_tranlsated = translate(text, from_translate, to_translate)
        return (text_tranlsated,)

    @classmethod
    def VALIDATE_INPUTS(cls, from_translate, to_translate, text):
        return True

```
