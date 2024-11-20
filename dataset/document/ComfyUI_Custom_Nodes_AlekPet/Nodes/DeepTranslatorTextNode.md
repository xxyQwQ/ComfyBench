# Deep Translator Text Node
## Documentation
- Class name: `DeepTranslatorTextNode`
- Category: `AlekPet Nodes/text`
- Output node: `False`

The DeepTranslatorTextNode is designed for advanced text translation tasks, leveraging deep learning models to enhance translation accuracy and support for multiple languages. It extends the capabilities of a standard translation node by incorporating deep learning techniques to improve the quality of translations, especially in complex or nuanced text scenarios.
## Input types
### Required
- **`from_translate`**
    - Specifies the source language from which the text will be translated, crucial for determining the translation context and accuracy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`to_translate`**
    - Specifies the target language to which the text will be translated, essential for ensuring the translation meets the desired linguistic and cultural nuances.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`add_proxies`**
    - Indicates whether proxies should be added to the translation request, affecting the node's ability to bypass geo-restrictions or network limitations.
    - Comfy dtype: `[[True, False]]`
    - Python dtype: `bool`
- **`proxies`**
    - Provides proxy details for the translation request, enabling the node to operate in environments with restricted internet access.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`auth_data`**
    - Contains authentication data required by certain translation services, ensuring secure access to premium translation features.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`service`**
    - Specifies the translation service to be used, impacting the translation's quality, speed, and language support.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text`**
    - The text input to be translated, highlighting the node's core functionality in processing and translating textual content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The translated text, demonstrating the node's ability to accurately convey the original message in a different language.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewTextNode](../../ComfyUI_Custom_Nodes_AlekPet/Nodes/PreviewTextNode.md)
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)



## Source code
```python
class DeepTranslatorTextNode(DeepTranslatorCLIPTextEncodeNode):
    @classmethod
    def INPUT_TYPES(self):
        types = super().INPUT_TYPES()
        del types["required"]["clip"]
        return types

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "deep_translate_text"

    CATEGORY = "AlekPet Nodes/text"

    def deep_translate_text(
        self,
        from_translate,
        to_translate,
        add_proxies,
        proxies,
        auth_data,
        service,
        text,
    ):
        # Select service
        service = service_correct_reg.sub("", service)
        if self.current_service != service:
            self.langs_support = selectService(service)
            self.current_service = service

        # Translate
        text_tranlsated = deep_translator_function(
            from_translate,
            to_translate,
            add_proxies,
            proxies,
            auth_data,
            service,
            text,
            self.langs_support,
        )

        return (text_tranlsated,)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        from_translate,
        to_translate,
        add_proxies,
        proxies,
        auth_data,
        service,
        text,
    ):
        # lang, is_support, detect = isset_languages(text, service, from_translate, self.langs_support)
        # if not is_support:
        #     return f"[Deep Translator] Service \"{service}\", no support for the provided language: {detect}!"
        return True

```
