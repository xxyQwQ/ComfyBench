# ∞ EmptyAgentsList
## Documentation
- Class name: `EmptyAgentsList`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

The EmptyAgentsList node is designed to initialize an empty list of agents. This functionality is essential for scenarios where a fresh start is needed or when beginning to populate a list of agents from scratch.
## Input types
### Required
## Output types
- **`agents`**
    - Comfy dtype: `AGENTS`
    - Returns an empty list, signifying the initialization of an agent list without any pre-existing agents.
    - Python dtype: `List[Agent]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class EmptyAgentsList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("AGENTS",)
    RETURN_NAMES = ("agents",)

    FUNCTION = "create_list"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

    def create_list(self):
        return ([],)

```
