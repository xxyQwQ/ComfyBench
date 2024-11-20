---
tags:
- Agents
- LLM
- LLMChat
---

# ∞ Conversable Agent (Adv)
## Documentation
- Class name: `ConversableAgentCreatorAdvanced`
- Category: `SALT/Language Toolkit/Agents`
- Output node: `False`

This node facilitates the advanced creation of conversable agents, allowing for the customization of agent attributes such as name, system message, and optional parameters like LLM model, default auto-reply, and a description. It's designed to enable the creation of sophisticated conversational AI agents that can interact with users, process document QA tasks, and provide automated responses.
## Input types
### Required
- **`name`**
    - The name of the conversable agent being created. It serves as a unique identifier and a way to reference the agent in interactions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`system_message`**
    - A system message that defines the agent's initial instructions or guidelines, setting the context for its operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`llm_model`**
    - An optional parameter specifying the language learning model to be used by the agent, enhancing its conversational capabilities.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `dict`
- **`default_auto_reply`**
    - The default message the agent will use when it cannot generate a response based on the LLM model or code execution.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`description`**
    - A brief description of the agent, which can be used for identification or informational purposes by other agents.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`agent`**
    - Comfy dtype: `AGENT`
    - The conversable agent created by this node, ready for interaction and capable of processing document QA tasks.
    - Python dtype: `ConversableAgent`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConversableAgentCreatorAdvanced:
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"name": ("STRING", {"multiline": False, "placeholder": "Assistant"}),
				"system_message": ("STRING", {
					"multiline": True,
					"default": "You are a helpful AI assistant. You can help with document QA. Return 'TERMINATE' when the task is done."
				}),
			},
			"optional": {
				"llm_model": ("LLM_MODEL",),
				# default auto reply when no code execution or llm-based reply is generated.
				"default_auto_reply": ("STRING", {"multiline": True}),
				# a short description of the agent, this description is used by other agents.
				"description": ("STRING", {"multiline": True}),
			}
		}

	RETURN_TYPES = ("AGENT",)
	RETURN_NAMES = ("agent",)

	FUNCTION = "create_agent"
	CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

	def create_agent(self, name, system_message, llm_model=None, default_auto_reply="", description=None):
		agent = ConversableAgent(
			name=name,
			system_message=system_message,
			llm_config={"config_list": [{"model": llm_model["llm"].model, "api_key": llm_model["llm"].api_key}]} if llm_model is not None else False,
			human_input_mode="NEVER",
			default_auto_reply=default_auto_reply,
			description=description if description is not None else system_message,
		)
		return (agent,)

```
