# ∞ AddAgentToList
## Documentation
- Class name: `AddAgentToList`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

The `AddAgentToList` node is designed to add a specified agent to an existing list of agents, facilitating dynamic management of agent collections within a chat or agent-based system.
## Input types
### Required
- **`agent`**
    - The `agent` parameter represents the individual agent to be added to the list. It is crucial for expanding the agent list dynamically.
    - Comfy dtype: `AGENT`
    - Python dtype: `object`
- **`agents`**
    - The `agents` parameter is the current list of agents to which the new agent will be added. It serves as the foundation for dynamically managing agent collections.
    - Comfy dtype: `AGENTS`
    - Python dtype: `list[object]`
## Output types
- **`agents`**
    - Comfy dtype: `AGENTS`
    - Returns the updated list of agents after adding the specified agent, reflecting the new state of the agent collection.
    - Python dtype: `list[object]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AddAgentToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "agent": ("AGENT",),
                "agents": ("AGENTS",),
            },
        }

    RETURN_TYPES = ("AGENTS",)
    RETURN_NAMES = ("agents",)

    FUNCTION = "add_agent"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

    def add_agent(self, agent, agents):
        agents.append(agent)
        return (agents,)

```
