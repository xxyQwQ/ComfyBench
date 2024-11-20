---
tags:
- Model
- ModelList
---

# Get Models
## Documentation
- Class name: `Get Models`
- Category: `Bmad/dump`
- Output node: `True`

The Get Models node is designed to facilitate the retrieval and organization of various model-related files within a specified output directory. It supports a flexible selection mechanism, allowing users to specify which types of model files (e.g., checkpoints, configurations, embeddings) they wish to dump, either individually or collectively, into a structured JSON file. This functionality aids in managing and cataloging model assets for easier access and reference.
## Input types
### Required
- **`dump`**
    - Specifies the type of model-related files to retrieve. It allows for a broad selection, including options like 'all models' for a comprehensive dump or specific categories such as 'checkpoints' or 'embeddings', tailoring the output to user needs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetModels:
    dump_option = ['all models',
                   'checkpoints',
                   'clip',
                   'clip_vision',
                   'configs',
                   'controlnet',
                   'diffusers',
                   'embeddings',
                   'gligen',
                   'hypernetworks',
                   'loras',
                   'style_models',
                   'upscale_models',
                   'vae'
                   ]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dump": (cls.dump_option, {"default": "all models"})
        }
        }

    RETURN_TYPES = ()
    FUNCTION = "dump_it"
    CATEGORY = f"{base_category_path}/dump"
    OUTPUT_NODE = True

    def dump_it(self, dump):
        dump_data = {}

        if dump == 'all models':
            for item in self.dump_option[1:]:
                dump_data[item] = folder_paths.get_filename_list(item)
        else:
            dump_data['list'] = folder_paths.get_filename_list(dump)

        file = f"{dump}.json"
        file = os.path.join(self.output_dir, file)
        with open(file, 'w') as f:
            json.dump(dump_data, f, indent=1)

        return ()

```
