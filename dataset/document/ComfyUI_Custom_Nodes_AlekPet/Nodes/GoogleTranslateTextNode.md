# Google Translate Text Node
## Documentation
- Class name: `GoogleTranslateTextNode`
- Category: `AlekPet Nodes/text`
- Output node: `False`

This node is designed to translate text from one language to another, offering an option for manual translation bypass. It leverages the Google Translate API or a manual override to return the translated text, streamlining the process of language translation within workflows.
## Input types
### Required
- **`from_translate`**
    - Specifies the source language of the text to be translated, or 'auto' to automatically detect the language. This parameter is crucial for determining the starting point of translation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Union[str, List[str]]`
- **`to_translate`**
    - Defines the target language for the translation, guiding the node to convert the input text into the desired language.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`manual_translate`**
    - A boolean flag that, when set to True, bypasses the translation process and returns the original text. This allows for optional manual control over the translation.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`text`**
    - The text to be translated, serving as the primary input for the translation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The translated text, either through the Google Translate API or as manually inputted, depending on the manual_translate flag.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GoogleTranslateTextNode(GoogleTranslateCLIPTextEncodeNode):

    @classmethod
    def INPUT_TYPES(self):
        return_types = super().INPUT_TYPES()
        del return_types["required"]["clip"]
        return return_types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "translate_text"

    CATEGORY = "AlekPet Nodes/text"

    def translate_text(self, **kwargs):
        from_translate = kwargs.get("from_translate")
        to_translate = kwargs.get("to_translate")
        manual_translate = kwargs.get("manual_translate", False)
        text = kwargs.get("text")

        text_tranlsated = (
            translate(text, from_translate, to_translate)
            if not manual_translate
            else text
        )
        return (text_tranlsated,)

```
