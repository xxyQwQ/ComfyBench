---
tags:
- Prompt
---

# Get Prompt
## Documentation
- Class name: `Get Prompt`
- Category: `Bmad/dump`
- Output node: `True`

The Get Prompt node is designed to process and modify a given prompt structure for execution or output purposes. It performs deep copying to avoid altering the original prompt, cleanses the prompt by removing unnecessary widget inputs, and supports outputting the modified prompt either to the console or to a file based on specified parameters. This node plays a crucial role in preparing prompts for execution by ensuring they are in the correct format and contain only the necessary information.
## Input types
### Required
- **`api_prompt`**
    - Specifies the output mode for the modified prompt, influencing how the prompt is processed and presented.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetPrompt:
    prompt_mode = ["print to console", "save to file"]

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "api_prompt": (cls.prompt_mode, {"default": cls.prompt_mode[0]})
        },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "getPrompt"
    CATEGORY = f"{base_category_path}/dump"
    OUTPUT_NODE = True

    def getPrompt(self, api_prompt, prompt, unique_id):
        # changing the original will mess the prompt execution, therefore make a copy
        prompt = copy.deepcopy(prompt)

        # remove this node from the prompt
        this_node = prompt[unique_id]
        del prompt[unique_id]

        # remove widgtes inputs from RequestInputs, only "values" is needed.
        for key in prompt:
            if prompt[key]["class_type"] == "RequestInputs":
                inputs = prompt[key]["inputs"]
                for attribute in list(inputs.keys()):
                    if attribute != "values":
                        del inputs[attribute]
                break  # supposes only 1 RequestInputs node exists

        prompt = {"prompt": prompt}

        # print to console or file
        if api_prompt == "print to console":
            print(json.dumps(prompt))
        elif api_prompt == "save to file":
            # TODO
            # avoid collisions (maybe just name it w/ date/time prefix?)
            # instead of owerriding the file
            file = "prompt.json"
            file = os.path.join(self.output_dir, file)
            with open(file, 'w') as f:
                json.dump(prompt, f, indent=1)

        return ()

```
