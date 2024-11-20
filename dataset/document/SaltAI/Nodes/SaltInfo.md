---
tags:
- Audio
- SaltNodes
---

# Salt Workflow Info
## Documentation
- Class name: `SaltInfo`
- Category: `SALT/IO`
- Output node: `True`

The SaltInfo node is designed to log and return basic information about a workflow, including its title and description. It serves as a means to encapsulate and convey workflow metadata, facilitating easier identification and understanding of the workflow's purpose.
## Input types
### Required
- **`workflow_title`**
    - The title of the workflow, serving as a concise identifier and summary of the workflow's objective.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`workflow_description`**
    - A detailed description of the workflow, outlining its purpose, functionality, and any other relevant information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`title`**
    - Comfy dtype: `STRING`
    - The title of the workflow, as provided in the input.
    - Python dtype: `str`
- **`description`**
    - Comfy dtype: `STRING`
    - The detailed description of the workflow, as provided in the input.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltInfo:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_title": ("STRING", {}),
                "workflow_description": ("STRING", {}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("title", "description")

    FUNCTION = "info"
    CATEGORY = "SALT/IO"

    def info(self, workflow_title, workflow_description, unique_id=0):

        logger.info(f"[SaltInfo_{unique_id}] Workflow Info:")
        logger.info(f"Title: {workflow_title}")
        logger.info(f"Description: {workflow_description}")

        return (workflow_title, workflow_description)

```
