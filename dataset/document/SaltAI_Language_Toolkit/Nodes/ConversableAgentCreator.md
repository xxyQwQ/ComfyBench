---
tags:
- Agents
- LLM
- LLMChat
---

# ∞ Conversable Agent
## Documentation
- Class name: `ConversableAgentCreator`
- Category: `SALT/Language Toolkit/Agents`
- Output node: `False`

This node is designed to facilitate the creation of conversable agents, which are AI entities capable of engaging in dialogue based on a predefined system message. It abstracts the complexities of configuring such agents, allowing users to specify basic parameters like the agent's name and its operational message, optionally integrating a language model for enhanced interaction capabilities.
## Input types
### Required
- **`name`**
    - Specifies the name of the conversable agent to be created, serving as a unique identifier and a way to reference the agent in interactions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`system_message`**
    - Defines the initial message or instruction that the agent will use to guide its interactions, setting the tone and scope of its conversational abilities.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`llm_model`**
    - An optional parameter that allows for the integration of a language learning model to enhance the agent's conversational capabilities, providing a more dynamic and responsive interaction experience.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `dict`
## Output types
- **`agent`**
    - Comfy dtype: `AGENT`
    - The created conversable agent, ready to be utilized for engaging in dialogues and performing tasks as defined by its configuration.
    - Python dtype: `ConversableAgent`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConversableAgentCreator:
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
			}
		}

	RETURN_TYPES = ("AGENT",)
	RETURN_NAMES = ("agent",)

	FUNCTION = "create_agent"
	CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents"

	def create_agent(self, name, system_message, llm_model=None):
		agent = ConversableAgent(
			name=name,
			system_message=system_message,
			llm_config={"config_list": [{"model": llm_model["llm"].model, "api_key": llm_model["llm"].api_key}]} if llm_model is not None else False,
			human_input_mode="NEVER",
		)
		return (agent,)

```
