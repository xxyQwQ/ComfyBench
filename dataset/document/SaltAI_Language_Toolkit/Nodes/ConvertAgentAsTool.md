# ∞ Agent As Tool
## Documentation
- Class name: `ConvertAgentAsTool`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

This node is designed to transform an agent into a tool that can be utilized within a broader system. It encapsulates the functionality of an agent, allowing it to be invoked as a callable tool for processing messages.
## Input types
### Required
- **`agent`**
    - The agent parameter represents the conversational agent to be converted into a tool. This conversion enables the agent's capabilities to be accessed in a tool-like manner, facilitating its integration into larger workflows or systems.
    - Comfy dtype: `AGENT`
    - Python dtype: `ConversableAgent`
## Output types
- **`tool`**
    - Comfy dtype: `TOOL`
    - The output is a tool encapsulating the agent's functionality, enabling it to be called with messages and return responses. This allows the agent's conversational capabilities to be utilized in a broader context.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConvertAgentAsTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "agent": ("AGENT",),
            },
        }

    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ("tool",)

    FUNCTION = "create_tool"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Tools"

    def create_tool(self, agent):
        agent = clone_conversable_agent(agent)
        def agent_call(message: Annotated[str, "Message to ask the assistant."]):
            return agent.generate_reply(messages=[{"content": message, "role": "user"}])

        tool = {
            "name": "agent_call",
            "description": agent.description,
            "function": agent_call,
        }
        return (tool,)

```
