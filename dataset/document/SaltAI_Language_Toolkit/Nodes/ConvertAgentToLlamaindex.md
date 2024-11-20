---
tags:
- Agents
- LLM
- LLMChat
---

# ∞ Convert Agent To Llamaindex
## Documentation
- Class name: `ConvertAgentToLlamaindex`
- Category: `SALT/Language Toolkit/Agents`
- Output node: `False`

This node is designed to transform an existing agent into a format compatible with LlamaIndex, enabling its integration and utilization within the LlamaIndex ecosystem. It allows for the extension of an agent's capabilities by optionally incorporating a pre-defined embedding model.
## Input types
### Required
- **`agent`**
    - The agent to be converted. This is the primary input that specifies the agent whose functionalities are to be made compatible with the LlamaIndex format.
    - Comfy dtype: `AGENT`
    - Python dtype: `Custom object representing an agent, specific to the system's architecture`
### Optional
- **`optional_embed_model`**
    - An optional embedding model that can be included to enhance the agent's capabilities within the LlamaIndex ecosystem.
    - Comfy dtype: `LLM_EMBED_MODEL`
    - Python dtype: `Custom object representing a LlamaIndex embedding model, specific to the system's architecture`
## Output types
- **`model`**
    - Comfy dtype: `LLM_MODEL`
    - The transformed agent, now in a format that is compatible with LlamaIndex, ready for integration and utilization.
    - Python dtype: `Custom object representing a LlamaIndex model, specific to the system's architecture`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConvertAgentToLlamaindex:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"agent": ("AGENT",),
			},
			"optional": {
				"optional_embed_model": ("LLM_EMBED_MODEL",)
            }
		}

	RETURN_TYPES = ("LLM_MODEL",)
	RETURN_NAMES = ("model",)

	FUNCTION = "convert_agent"
	CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

	def convert_agent(self, agent, optional_embed_model=None):
		llm = {"llm": BaseModel(agent)}
		if optional_embed_model:
			llm.update(optional_embed_model)
		return (llm,)

```
