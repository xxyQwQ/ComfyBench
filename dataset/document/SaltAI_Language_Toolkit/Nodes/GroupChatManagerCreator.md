# ∞ Group Chat Manager
## Documentation
- Class name: `GroupChatManagerCreator`
- Category: `SALT/Language Toolkit/Agents/Chat`
- Output node: `False`

The GroupChatManagerCreator node specializes in creating a group chat manager agent designed to oversee and manage group chat interactions among multiple agents. It is tailored for scenarios where group chat functionality is essential, providing tools to configure and control the chat environment effectively.
## Input types
### Required
- **`name`**
    - The 'name' parameter specifies the name of the group chat manager. It is a key identifier for the chat manager within the group chat environment.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`system_message`**
    - The 'system_message' parameter is a customizable message that the group chat manager can use to communicate system-level information within the group chat.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`llm_model`**
    - The 'llm_model' parameter allows for the specification of a language model to be used by the group chat manager, enabling advanced language understanding and response generation capabilities.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `dict`
- **`max_consecutive_auto_reply`**
    - This parameter sets the maximum number of consecutive automatic replies the group chat manager can send, helping to prevent spam and maintain conversation quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`group_manager`**
    - Comfy dtype: `GROUP_MANAGER`
    - The output is a configured group manager agent, ready to be deployed in managing group chat interactions.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GroupChatManagerCreator:
	"""
	A chat manager agent that can manage a group chat of multiple agents.
	Can only be used in group chats.
	"""
	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"name": ("STRING", {"multiline": False, "placeholder": "Manager"}),
				"system_message": ("STRING", {
					"multiline": True,
					"placeholder": "Group chat manager.",
				}),
			},
			"optional": {
				"llm_model": ("LLM_MODEL",),
				"max_consecutive_auto_reply": ("INT", {"default": 10}),
			}
		}

	RETURN_TYPES = ("GROUP_MANAGER",)
	RETURN_NAMES = ("group_manager",)

	FUNCTION = "create_agent"
	CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Chat"

	def create_agent(self, name, system_message, llm_model=None, max_consecutive_auto_reply=None):
		group_manager = {
			"name": name,
			"system_message": system_message,
			"llm_config": {
				"config_list": [
					{
						"model": llm_model["llm"].model,
						"api_key": llm_model["llm"].api_key,
					}
				]
			} if llm_model is not None else None,
			"max_consecutive_auto_reply": max_consecutive_auto_reply,
		}
		return (group_manager,)

```
