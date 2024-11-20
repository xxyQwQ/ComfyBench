---
tags:
- CLIP
- CLIPConditioning
- Conditioning
---

# Deep Translator CLIP Text Encode Node
## Documentation
- Class name: `DeepTranslatorCLIPTextEncodeNode`
- Category: `AlekPet Nodes/conditioning`
- Output node: `False`

This node is designed to encode text inputs using a deep translation mechanism combined with CLIP text encoding, facilitating the translation of text into different languages before encoding it into a format suitable for conditioning models. It serves as a foundational component for nodes that require text translation and encoding capabilities for further processing or model conditioning.
## Input types
### Required
- **`from_translate`**
    - Specifies the source language of the text to be translated. It plays a crucial role in determining the initial language from which the text will be translated, affecting the accuracy and relevance of the translation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`to_translate`**
    - Defines the target language to which the text will be translated. This parameter is essential for guiding the translation process towards the desired language, impacting the final output's applicability for specific linguistic contexts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`add_proxies`**
    - Indicates whether proxies should be added to the translation request, enabling the bypass of potential network restrictions.
    - Comfy dtype: `[[True, False]]`
    - Python dtype: `bool`
- **`proxies`**
    - Specifies the proxy settings to be used if 'add_proxies' is true, facilitating the translation request through specified network paths.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`auth_data`**
    - Authentication data required for accessing the translation service, ensuring secure and authorized translation requests.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`service`**
    - The translation service to be used, such as 'GoogleTranslator' or another supported service, determining the translation mechanism and capabilities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`text`**
    - The text input to be translated and encoded. This parameter is central to the node's functionality, as it provides the raw content for translation and subsequent encoding.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A CLIP model instance used for text encoding. It is crucial for converting the translated text into a format compatible with conditioning models, enhancing the text's utility for model inputs.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The encoded representation of the translated text, suitable for use as conditioning input in various models. It includes both the conditioning vectors and pooled output for comprehensive model conditioning.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
- **`string`**
    - Comfy dtype: `STRING`
    - The translated text, providing a direct linguistic output of the translation process. It serves as a readable representation of the text after translation, useful for verification or further processing.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DeepTranslatorCLIPTextEncodeNode:

    @classmethod
    def INPUT_TYPES(self):
        self.current_service = ""
        self.proxies = ""
        self.auth_data = ""

        self.langs_support = selectService("GoogleTranslator")
        langs_support = list(self.langs_support.keys())

        return {
            "required": makeRequiredFields(langs_support),
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    FUNCTION = "deep_translate_text"

    CATEGORY = "AlekPet Nodes/conditioning"

    def deep_translate_text(
        self,
        from_translate,
        to_translate,
        add_proxies,
        proxies,
        auth_data,
        service,
        text,
        clip,
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

        tokens = clip.tokenize(text_tranlsated)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return (
            [[cond, {"pooled_output": pooled}]],
            text_tranlsated,
        )

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
        clip,
    ):
        # lang, is_support, detect = isset_languages(text, service, from_translate, self.langs_support)
        # if not is_support:
        #     return f"[Deep Translator] Service \"{service}\", no support for the provided language: {detect}!"
        return True

```
