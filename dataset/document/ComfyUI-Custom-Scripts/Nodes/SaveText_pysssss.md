# Save Text 🐍
## Documentation
- Class name: `SaveText_pysssss`
- Category: `utils`
- Output node: `True`

The SaveText node is designed for writing text to a file, with options to append to an existing file or create a new one. It abstracts file handling operations, allowing users to easily save text data without dealing with low-level file system details.
## Input types
### Required
- **`root_dir`**
    - Specifies the directory where the file will be saved. It's crucial for determining the file's path and ensuring the text is saved in the correct location.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`file`**
    - The name of the file where the text will be saved. This parameter works in conjunction with 'root_dir' to define the full path of the target file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`append`**
    - Controls whether the text should be appended to an existing file or if a new file should be created. It supports options like 'append' and 'new only'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`insert`**
    - Determines if a newline should be inserted before appending text when the 'append' option is used.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`text`**
    - The text content to be saved into the file. This is the primary data being written.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output is the text that was successfully written to the file.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaveText(TextFileNode):
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "root_dir": (list(get_valid_dirs()), {}),
                "file": ("STRING", {"default": "file.txt"}),
                "append": (["append", "overwrite", "new only"], {}),
                "insert": ("BOOLEAN", {
                    "default": True, "label_on": "new line", "label_off": "none",
                    "pysssss.binding": [{
                        "source": "append",
                        "callback": [{
                            "type": "if",
                            "condition": [{
                                "left": "$source.value",
                                "op": "eq",
                                "right": '"append"'
                            }],
                            "true": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": False
                            }],
                            "false": [{
                                "type": "set",
                                "target": "$this.disabled",
                                "value": True
                            }],
                        }]
                    }]
                }),
                "text": ("STRING", {"forceInput": True, "multiline": True})
            },
        }

    FUNCTION = "write_text"

    def write_text(self, **kwargs):
        self.file = get_file(kwargs["root_dir"], kwargs["file"])
        if kwargs["append"] == "new only" and os.path.exists(self.file):
            raise FileExistsError(
                self.file + " already exists and 'new only' is selected.")
        with open(self.file, "a+" if kwargs["append"] == "append" else "w") as f:
            is_append = f.tell() != 0
            if is_append and kwargs["insert"]:
                f.write("\n")
            f.write(kwargs["text"])

        return super().load_text(**kwargs)

```
