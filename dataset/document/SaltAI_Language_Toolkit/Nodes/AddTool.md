# ∞ Add Tool
## Documentation
- Class name: `AddTool`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

Facilitates the dynamic addition of tools to conversational agents by cloning and registering functions within these agents. This process enhances the agents' capabilities, allowing them to perform additional tasks as specified by the tool being added.
## Input types
### Required
- **`tool`**
    - Represents the tool to be added, including its function, name, and description. This parameter is crucial for defining the new capabilities to be integrated into the agent.
    - Comfy dtype: `TOOL`
    - Python dtype: `Dict[str, Any]`
- **`assistant`**
    - The conversational agent to which the tool will be added. This agent is cloned to ensure modifications do not affect the original instance.
    - Comfy dtype: `AGENT`
    - Python dtype: `AGENT`
### Optional
- **`executor`**
    - An optional agent responsible for executing the tool's function. If not provided, the assistant itself is used as the executor.
    - Comfy dtype: `AGENT`
    - Python dtype: `Optional[AGENT]`
## Output types
- **`assistant`**
    - Comfy dtype: `AGENT`
    - The modified conversational agent with the new tool added.
    - Python dtype: `AGENT`
- **`executor`**
    - Comfy dtype: `AGENT`
    - The agent designated to execute the added tool's function, which may be the same as the assistant if no executor is specified.
    - Python dtype: `AGENT`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AddTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tool": ("TOOL",),
                "assistant": ("AGENT",),
            },
            "optional": {
                "executor": ("AGENT",),
            }
        }

    RETURN_TYPES = ("AGENT", "AGENT",)
    RETURN_NAMES = ("assistant", "executor",)

    FUNCTION = "create_tool"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Tools"

    def create_tool(self, tool, assistant, executor=None):
        assistant = clone_conversable_agent(assistant)
        if executor is None:
            executor = assistant
        # Register the calculator function to the two agents.
        register_function(
            tool["function"],
            caller=assistant,
            executor=executor,
            name=tool["name"],
            description=tool["description"],
        )
        return (assistant, executor,)

```
