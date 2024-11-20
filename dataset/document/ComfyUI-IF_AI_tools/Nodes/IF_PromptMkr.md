---
tags:
- Prompt
---

# IF Prompt to Prompt💬
## Documentation
- Class name: `IF_PromptMkr`
- Category: `ImpactFrames💥🎞️`
- Output node: `False`

The IF_PromptMkr node is designed to transform an initial text prompt into a more detailed or stylistically altered prompt, leveraging various AI models and customizations. It enriches the input prompt with embellishments, styles, or negations based on user-selected options, aiming to generate more impactful or contextually appropriate prompts for further processing or creative generation.
## Input types
### Required
- **`input_prompt`**
    - The primary text prompt to be transformed. It serves as the base content for generating enriched or stylistically altered prompts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`base_ip`**
    - The IP address of the server where the AI model is hosted, used for sending requests to generate the enriched prompt.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`port`**
    - The port number on the server for accessing the AI model, used in conjunction with the base IP address.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`engine`**
    - Specifies the AI model engine to use for prompt generation, allowing selection from options like 'ollama', 'openai', or 'anthropic'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`selected_model`**
    - The specific AI model selected for generating the prompt, determined dynamically based on the engine, base IP, and port.
    - Comfy dtype: `[]`
    - Python dtype: `tuple()`
- **`profile`**
    - The profile setting that influences the generation style or approach, selected from a predefined list of profiles.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`embellish_prompt`**
    - A selection of embellishments to apply to the input prompt, enhancing its detail or thematic elements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`style_prompt`**
    - A selection of styles to apply to the input prompt, altering its presentation or tone.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`neg_prompt`**
    - A selection of negations to apply to the input prompt, introducing contrast or oppositional elements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list[str]`
- **`temperature`**
    - Controls the creativity or randomness of the generated prompt, with a range from 0.0 (less random) to 1.0 (more random).
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`max_tokens`**
    - The maximum number of tokens to generate for the enriched prompt, setting an upper limit on its length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for the random number generator, ensuring reproducibility of the generated prompt when set.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`random`**
    - A boolean flag that, when true, uses the seed value for generating the prompt, otherwise uses the temperature setting.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`keep_alive`**
    - A boolean flag that, when true, keeps the AI model loaded between requests, potentially improving performance.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`Question`**
    - Comfy dtype: `STRING`
    - The original input prompt, returned unchanged.
    - Python dtype: `str`
- **`Response`**
    - Comfy dtype: `STRING`
    - The enriched or stylistically altered prompt generated based on the input and selected options.
    - Python dtype: `str`
- **`Negative`**
    - Comfy dtype: `STRING`
    - The negated version of the input prompt, incorporating contrast or oppositional elements.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IFPrompt2Prompt: 
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response", "Negative",)
    FUNCTION = "sample"
    OUTPUT_NODE = False
    CATEGORY = "ImpactFrames💥🎞️"

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "input_prompt": ("STRING", {"multiline": True, "default": "Ancient mega-structure, small lone figure in the foreground"}),
                "base_ip": ("STRING", {"default": node.base_ip}),
                "port": ("STRING", {"default": node.port}),
                "engine": (["ollama", "openai", "anthropic"], {"default": node.engine}),
                #"selected_model": (node.get_models("node.engine", node.base_ip, node.port), {}), 
                "selected_model": ((), {}),
                "profile": ([name for name in node.profiles.keys()], {"default": node.profile}),
                "embellish_prompt": ([name for name in node.embellish_prompts.keys()], {}),
                "style_prompt": ([name for name in node.style_prompts.keys()], {}),
                "neg_prompt": ([name for name in node.neg_prompts.keys()], {}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature"}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps_Model", "label_off": "Unloads_Model"}),
            },
            "hidden": {
                "model": ("STRING", {"default": ""}),
            },
        }
    @classmethod
    def IS_CHANGED(cls, engine, base_ip, port, profile, keep_alive):
        node = cls()
        if engine != node.engine or base_ip != node.base_ip or port != node.port or node.selected_model != node.get_models(engine, base_ip, port) or keep_alive != node.keep_alive:
            node.engine = engine
            node.base_ip = base_ip
            node.port = port
            node.selected_model = node.get_models(engine, base_ip, port)
            node.profile = profile
            node.keep_alive = keep_alive
            return True
        return False

    def __init__(self):
        self.base_ip = "localhost" 
        self.port = "11434"     
        self.engine = "ollama" 
        self.selected_model = ""
        self.profile = "IF_PromptMKR"
        self.comfy_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.presets_dir = os.path.join(os.path.dirname(__file__), "presets")
        self.profiles_file = os.path.join(self.presets_dir, "profiles.json")
        self.profiles = self.load_presets(self.profiles_file)
        self.neg_prompts_file = os.path.join(self.presets_dir, "neg_prompts.json")
        self.embellish_prompts_file = os.path.join(self.presets_dir, "embellishments.json")
        self.style_prompts_file = os.path.join(self.presets_dir, "style_prompts.json")
        self.neg_prompts = self.load_presets(self.neg_prompts_file)
        self.embellish_prompts = self.load_presets(self.embellish_prompts_file)
        self.style_prompts = self.load_presets(self.style_prompts_file)
        
    def load_presets(self, file_path):
        with open(file_path, 'r') as f:
            presets = json.load(f)
        return presets
   
    def get_api_key(self, api_key_name, engine):
        if engine != "ollama":  
            api_key = os.getenv(api_key_name)
            if api_key:
                return api_key
        else:
            print(f'you are using ollama as the engine, no api key is required')

    def get_models(self, engine, base_ip, port):
        if engine == "ollama":
            api_url = f'http://{base_ip}:{port}/api/tags'
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                models = [model['name'] for model in response.json().get('models', [])]
                return models
            except Exception as e:
                print(f"Failed to fetch models from Ollama: {e}")
                return []
        elif engine == "anthropic":
            return ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        elif engine == "openai":
            return ["gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4-vision-preview", "gpt-4-1106-vision-preview", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"]
        else:
            print(f"Unsupported engine - {engine}")
            return []
              
    def sample(self, input_prompt, engine, base_ip, port, selected_model, embellish_prompt, style_prompt, neg_prompt, temperature, max_tokens, seed, random, keep_alive, profile):
        embellish_content = self.embellish_prompts.get(embellish_prompt, "")
        style_content = self.style_prompts.get(style_prompt, "")
        neg_content = self.neg_prompts.get(neg_prompt, "")
        profile_selected = self.profiles.get(profile, "")

        if engine == "anthropic":
            data = {
                'model': selected_model,
                'system': profile_selected ,
                'messages': [
                    {"role": "user", "content": input_prompt}
                ],
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        elif engine == "openai":
            if random == True:
                data = {
                    'model': selected_model, 
                    'messages': [
                        {"role": "system", "content": profile_selected },
                        {"role": "user", "content": input_prompt}
                    ],
                    'temperature': temperature,
                    'seed': seed,
                    'max_tokens': max_tokens  
                }
            else:
                data = {
                    'model': selected_model, 
                    'messages': [
                        {"role": "system", "content": profile_selected },
                        {"role": "user", "content": input_prompt}
                    ],
                    'temperature': temperature,
                    'max_tokens': max_tokens  
                }
        else:
            if random == True:
                data = {
                    "model": selected_model,
                    "system": profile_selected ,
                    "prompt": input_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "seed": seed,
                        "num_ctx": max_tokens,
                    },
                    "keep_alive": -1 if keep_alive else 0,
                }      
            else:
                data = {
                    "model": selected_model,
                    "system": profile_selected ,
                    "prompt": input_prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "seed": seed,
                        "num_ctx": max_tokens,
                    },
                    "keep_alive": -1 if keep_alive else 0,
                }

        generated_text = self.send_request(engine, base_ip, port, data, headers={"Content-Type": "application/json"})

        if generated_text:
            combined_prompt = f"{embellish_content} {generated_text} {style_content}"
            return input_prompt, combined_prompt, neg_content
        else:
            return None, None, None
              
    def send_request(self, engine, base_ip, port, data, headers):
        if engine == "ollama":
            api_url = f'http://{base_ip}:{port}/api/generate'
            response = requests.post(api_url, headers=headers, json=data)
            if response.status_code == 200:
                response_data = response.json()
                prompt_response = response_data.get('response', 'No response text found')
                
                # Ensure there is a response to construct the full description
                if prompt_response != 'No response text found':
                    return prompt_response
                else:
                    return "No valid response generated for the image."
            else:
                print(f"Failed to fetch response, status code: {response.status_code}")
                return "Failed to fetch response from Ollama."
        elif engine == "anthropic":
            anthropic_api_key = self.get_api_key("ANTHROPIC_API_KEY", engine)
            try:
                base_url = 'https://api.anthropic.com/v1/messages'
                anthropic_headers = {
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",  
                    "Content-Type": "application/json"
                }
                response = requests.post(base_url, headers=anthropic_headers, json=data)
                if response.status_code == 200:
                    messages = response.json().get('content', [])
                    generated_text = ''.join([msg.get('text', '') for msg in messages if msg.get('type') == 'text'])
                    return generated_text
                else:
                    print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                    return None
            except Exception as e:
                print(f"Error: Anthropic request failed - {e}")
                return None
        elif engine == "openai":
            openai_api_key = self.get_api_key("OPENAI_API_KEY", engine)
            try:
                base_url = 'https://api.openai.com/v1/chat/completions'
                openai_headers = {
                    "Authorization": f"Bearer {openai_api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(base_url, headers=openai_headers, json=data)
                if response.status_code == 200:
                    response_data = response.json()
                    print("Debug Response:", response_data)  
                    choices = response_data.get('choices', [])
                    if choices:
                        choice = choices[0]
                        messages = choice.get('message', {'content': ''})  
                        generated_text = messages.get('content', '') 
                        return generated_text
                    else:
                        print("No choices found in response")
                        return None
                else:
                    print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                    return None
            except Exception as e:
                print(f"Error: OpenAI request failed - {e}")
                return None, None, None

```
