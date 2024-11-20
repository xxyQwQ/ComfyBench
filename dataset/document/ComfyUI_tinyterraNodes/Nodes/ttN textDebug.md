---
tags:
- Debugging
---

# textDebug
## Documentation
- Class name: `ttN textDebug`
- Category: `🌏 tinyterra/text`
- Output node: `True`

The ttN textDebug node is designed for debugging purposes within the tinyterra text processing framework. It allows for the inspection and manipulation of text data, providing a means to evaluate and refine text processing workflows.
## Input types
### Required
- **`print_to_console`**
    - Determines whether the debugged text data should be printed to the console, facilitating real-time inspection.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`console_title`**
    - Specifies a title for the console output when 'print_to_console' is true, allowing for easier identification of the debugged text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`execute`**
    - Controls when the node executes, with options like 'Always' or 'On Change', to optimize debugging efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`text`**
    - The 'text' input is essential for providing the text data to be debugged. It supports multiline input and dynamic prompts, enabling flexible and interactive debugging sessions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The output 'text' is the processed or inspected text data, allowing for direct observation of the debugging results.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_textDebug:
    version = '1.0.'
    def __init__(self):
        self.num = 0

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "print_to_console": ([False, True],),
                    "console_title": ("STRING", {"default": ""}),
                    "execute": (["Always", "On Change"],),
                    "text": ("STRING", {"default": '', "multiline": True, "forceInput": True, "dynamicPrompts": True}),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           "ttNnodeVersion": ttN_textDebug.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "write"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/text"

    def write(self, print_to_console, console_title, execute, text, prompt, extra_pnginfo, my_unique_id):
        if execute == "Always":
            def IS_CHANGED(self):
                self.num += 1 if self.num == 0 else -1
                return self.num
            setattr(self.__class__, 'IS_CHANGED', IS_CHANGED)

        if execute == "On Change":
            if hasattr(self.__class__, 'IS_CHANGED'):
                delattr(self.__class__, 'IS_CHANGED')

        if print_to_console == True:
            if console_title != "":
                ttNl(text).t(f'textDebug[{my_unique_id}] - {CC.VIOLET}{console_title}').p()
            else:
                input_node = prompt[my_unique_id]["inputs"]["text"]

                input_from = None
                for node in extra_pnginfo["workflow"]["nodes"]:
                    if node['id'] == int(input_node[0]):
                        input_from = node['outputs'][input_node[1]].get('label')
                    
                        if input_from == None:
                            input_from = node['outputs'][input_node[1]].get('name')

                ttNl(text).t(f'textDebug[{my_unique_id}] - {CC.VIOLET}{input_from}').p()

        return {"ui": {"text": text},
                "result": (text,)}

```
