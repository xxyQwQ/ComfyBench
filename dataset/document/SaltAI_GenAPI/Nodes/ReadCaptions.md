# ReadCaptions
## Documentation
- Class name: `ReadCaptions`
- Category: `SALT/CLUSTERING`
- Output node: `False`

The ReadCaptions node is designed to extract captions from a specified JSON file, focusing on parsing and aggregating caption data related to various categories within the file. This node plays a crucial role in processing and preparing textual data for further analysis or processing steps, such as clustering or generating new content based on the extracted captions.
## Input types
### Required
- **`file_path`**
    - Specifies the path to the JSON file containing captions. This input is essential for locating and reading the file to extract captions, directly influencing the node's ability to perform its primary function.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`captions`**
    - Comfy dtype: `LIST`
    - A list of captions extracted and aggregated from the specified JSON file. This output is crucial for downstream tasks that require textual data for analysis, processing, or generation.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ReadCaptions:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("captions",)

    FUNCTION = "read_captions"
    CATEGORY = f"{MENU_NAME}/CLUSTERING"

    def read_captions(self, file_path):
        with open(file_path) as json_file:
            d = json.load(json_file)
        captions = []
        for cat in d:
            for k, v in d[cat].items():
                if "captions" in v:
                    captions.append(", ".join(json.loads(v)['captions']))
        return (captions,)

```
