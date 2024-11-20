---
tags:
- Prompt
---

# IF Image to Prompt🖼️
## Documentation
- Class name: `IF_ImagePrompt`
- Category: `ImpactFrames💥🎞️`
- Output node: `True`

The IF_ImagePrompt node transforms visual inputs into textual prompts, leveraging user profiles and image characteristics to generate tailored text descriptions. It supports customization through various parameters, including style and embellishment options, to enhance the generated prompts.
## Input types
### Required
- **`image`**
    - The image input can be a torch.Tensor, a PIL.Image, or a file path to an image. It serves as the primary visual content for generating the text prompt.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Union[torch.Tensor, PIL.Image, str]`
- **`image_prompt`**
    - A textual description or directive provided by the user to guide the generation of the text prompt, influencing its thematic direction.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`base_ip`**
    - The IP address of the server where the model engine is hosted, required for sending the image processing request.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`port`**
    - The port number on the server for accessing the model engine, used in conjunction with the base IP address.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`engine`**
    - The name of the engine (e.g., ollama, openai, anthropic) used for generating the text prompt, determining the underlying AI model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`selected_model`**
    - The specific model selected for generating the text prompt, chosen from the available models on the specified engine.
    - Comfy dtype: `[]`
    - Python dtype: `str`
- **`profile`**
    - A user profile identifier used to select specific settings or preferences for text generation, affecting the style and content of the generated prompts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`embellish_prompt`**
    - An optional parameter for adding embellishment to the generated text, enhancing its descriptive quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`style_prompt`**
    - A parameter for specifying the style of the generated text, affecting its tone and presentation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`neg_prompt`**
    - A parameter for providing a negative or contrasting prompt, used to generate alternative text descriptions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`temperature`**
    - Controls the creativity of the generated text. A higher temperature results in more varied text.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`max_tokens`**
    - The maximum number of tokens to generate for the text prompt, limiting its length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for the random number generator, ensuring reproducibility of the generated text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`random`**
    - A boolean parameter that, when true, uses a random seed for text generation, adding variability to the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`keep_alive`**
    - A boolean parameter indicating whether to keep the model loaded between requests, affecting resource utilization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`Question`**
    - Comfy dtype: `STRING`
    - A system-generated message or question based on the image prompt and user profile.
    - Python dtype: `str`
- **`Response`**
    - Comfy dtype: `STRING`
    - The main text description generated from the visual input, incorporating any specified style or embellishment.
    - Python dtype: `str`
- **`Negative`**
    - Comfy dtype: `STRING`
    - A negative or alternative text prompt, possibly used for contrast or to provide different perspectives.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IFImagePrompt:

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("Question", "Response", "Negative",)
    FUNCTION = "describe_picture"
    OUTPUT_NODE = True
    CATEGORY = "ImpactFrames💥🎞️"

    @classmethod
    def INPUT_TYPES(cls):
        node = cls()
        return {
            "required": {
                "image": ("IMAGE", ),
                "image_prompt": ("STRING", {"multiline": True, "default": ""}),
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
                "max_tokens": ("INT", {"default": 160, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "random": ("BOOLEAN", {"default": False, "label_on": "Seed", "label_off": "Temperature"}),
                "keep_alive": ("BOOLEAN", {"default": False, "label_on": "Keeps_Model", "label_off": "Unloads_Model"}),
            },
            "hidden": {
                "model": ("STRING", {"default": ""}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, engine, base_ip, port, keep_alive, profile):
        node = cls()
        if engine != node.engine or base_ip != node.base_ip or port != node.port or node.selected_model != node.get_models(engine, base_ip, port) or keep_alive != node.keep_alive or profile != profile:
            node.engine = engine
            node.base_ip = base_ip
            node.port = port
            node.selected_model = node.get_models(engine, base_ip, port)
            node.keep_alive = keep_alive
            node.profile = profile
            return True
        return False
    
    def __init__(self):
        self.base_ip = "localhost" 
        self.port = "11434"     
        self.engine = "ollama" 
        self.selected_model = ""
        self.profile = "IF_PromptMKR_IMG"
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

    def tensor_to_image(self, tensor):
        # Ensure tensor is on CPU
        tensor = tensor.cpu()
        # Normalize tensor 0-255 and convert to byte
        image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
        # Create PIL image
        image = Image.fromarray(image_np, mode='RGB')
        return image


    def prepare_messages(self, image_prompt, profile):
        profile_selected = self.profiles.get(profile, "")
        empty_image_prompt = "Make a visual prompt for the following Image:"
        filled_image_prompt = textwrap.dedent("""\
            Act as a visual prompt maker with the following guidelines:
            - Describe the image in vivid detail.
            - Break keywords by commas.
            - Provide high-quality, non-verbose, coherent, concise, and not superfluous descriptions.
            - Focus solely on the visual elements of the picture; avoid art commentaries or intentions.
            - Construct the prompt by describing framing, subjects, scene elements, background, aesthetics.
            - Limit yourself up to 7 keywords per component  
            - Be varied and creative.
            - Always reply on the same line, use around 100 words long. 
            - Do not enumerate or enunciate components.
            - Do not include any additional information in the response.                                                       
            The following is an illustartive example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship:
            'Epic, Cover Art, Full body shot, dynamic angle, A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy. Ciberpunk, dramatic lighthing, highly detailed. ' 
             """)
        if image_prompt.strip() == "":
            system_message = filled_image_prompt
            user_message = empty_image_prompt
        else:
            system_message = profile_selected 
            user_message = image_prompt

        return system_message, user_message


    def describe_picture(self, image, engine, selected_model, base_ip, port, image_prompt, embellish_prompt, style_prompt, neg_prompt, temperature, max_tokens, seed, random, keep_alive, profile):

        embellish_content = self.embellish_prompts.get(embellish_prompt, "")
        style_content = self.style_prompts.get(style_prompt, "")
        neg_content = self.neg_prompts.get(neg_prompt, "")
        
        # Check the type of the 'image' object
        if isinstance(image, torch.Tensor):
            # Convert the tensor to a PIL image
            pil_image = self.tensor_to_image(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str) and os.path.isfile(image):
            pil_image = Image.open(image)
        else:
            print(f"Invalid image type: {type(image)}. Expected torch.Tensor, PIL.Image, or file path.")
            return "Invalid image type"

        # Convert the PIL image to base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        available_models = self.get_models(engine, base_ip, port)
        if available_models is None or selected_model not in available_models:
            error_message = f"Invalid model selected: {selected_model} for engine {engine}. Available models: {available_models}"
            print(error_message)
            raise ValueError(error_message)

        system_message, user_message = self.prepare_messages(image_prompt, profile)

        try:
            generated_text = self.send_request(engine, selected_model, base_ip, port, base64_image, system_message, user_message, temperature, max_tokens, seed, random, keep_alive)
            description = f"{embellish_content} {generated_text} {style_content}".strip()
            return  image_prompt, description, neg_content
        except Exception as e:
            print(f"Exception occurred: {e}")
            return "Exception occurred while processing image."


    def send_request(self, engine, selected_model, base_ip, port, base64_image, system_message, user_message, temperature, max_tokens, seed, random, keep_alive): 
        if engine == "anthropic":
            anthropic_api_key = self.get_api_key("ANTHROPIC_API_KEY", engine)
            anthropic_headers = {
                "x-api-key": anthropic_api_key,
                "anthropic-version": "2023-06-01",  
                "Content-Type": "application/json"
            }

            data = {
                "model": selected_model,
                "system": system_message,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": base64_image
                                    }
                                },
                            {"type": "text", "text": user_message}
                        ]
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            api_url = 'https://api.anthropic.com/v1/messages'

            response = requests.post(api_url, headers=anthropic_headers, json=data)
            if response.status_code == 200:
                response_data = response.json()
                messages = response_data.get('content', [])
                generated_text = ''.join([msg.get('text', '') for msg in messages if msg.get('type') == 'text'])
                return generated_text
            else:
                print(f"Error: Request failed with status code {response.status_code}, Response: {response.text}")
                return "Failed to fetch response from Anthropic."
                
        elif engine == "openai":
            openai_api_key = self.get_api_key("OPENAI_API_KEY", engine)
            openai_headers = {
                "Authorization": f"Bearer {openai_api_key}",
                "Content-Type": "application/json"
            }
            if random == True:
                data = {
                    "model": selected_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_message
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{base64_image}"
                                }
                            ]
                        }
                    ],
                    "temperature": temperature,
                    "seed": seed,
                    "max_tokens": max_tokens
                }
            else:
                data = {
                    "model": selected_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_message
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{base64_image}"
                                }
                            ]
                        }
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

            api_url = 'https://api.openai.com/v1/chat/completions'
            response = requests.post(api_url, headers=openai_headers, json=data)
            if response.status_code == 200:
                response_data = response.json()
                print("Debug Response:", response_data)
                choices = response_data.get('choices', [])
                if choices:
                    choice = choices[0]
                    message = choice.get('message', {})
                    generated_text = message.get('content', '')
                    return generated_text
                else:
                    print("No valid choices in the response.")
                    print("Full response:", response.text)
                    return "No valid response generated for the image."
            else:
                print(f"Failed to fetch response, status code: {response.status_code}")
                print("Full response:", response.text)
                return "Failed to fetch response from OpenAI."

        else:
            api_url = f'http://{base_ip}:{port}/api/generate'
            if random == True:
                data = {
                    "model": selected_model,
                    "system": system_message,
                    "prompt": user_message,
                    "stream": False,
                    "images": [base64_image],
                    "options": {
                        "temperature": temperature,
                        "seed": seed,
                        "num_ctx": max_tokens
                    },
                    "keep_alive": -1 if keep_alive else 0,
                }
            else:
                data = {
                    "model": selected_model,
                    "system": system_message,
                    "prompt": user_message,
                    "stream": False,
                    "images": [base64_image],
                    "options": {
                        "temperature": temperature,
                        "num_ctx": max_tokens,
                    },
                    "keep_alive": -1 if keep_alive else 0,
                }

            ollama_headers = {"Content-Type": "application/json"}
            response = requests.post(api_url, headers=ollama_headers, json=data)
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

```
