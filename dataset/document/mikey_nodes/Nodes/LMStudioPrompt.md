---
tags:
- Prompt
---

# LM Studio Prompt (Mikey)
## Documentation
- Class name: `LMStudioPrompt`
- Category: `Mikey/AI`
- Output node: `True`

LMStudioPrompt is designed to interface with the LM Studio API, offering a streamlined way to generate prompts by leveraging the capabilities of this specific API. It mirrors the functionality of a similar node but specifically utilizes LM Studio's services for prompt generation, emphasizing its integration with LM Studio's unique features and API structure.
## Input types
### Required
- **`input_prompt`**
    - The 'input_prompt' parameter represents the initial prompt or question provided by the user, serving as the basis for generating contextually relevant responses through the LM Studio API.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`mode`**
    - The 'mode' parameter determines the operational mode of the LM Studio API, potentially affecting the style or approach of the generated responses.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`custom_history`**
    - The 'custom_history' parameter allows for the inclusion of a custom interaction history, enhancing the context for the generated response by the LM Studio API.
    - Comfy dtype: `STRING`
    - Python dtype: `List[Dict[str, str]]`
- **`server_address`**
    - The 'server_address' parameter specifies the network address of the LM Studio server, required for API communication.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`server_port`**
    - The 'server_port' parameter indicates the network port of the LM Studio server, essential for establishing a connection.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - The 'seed' parameter is used to initialize the random number generator within the LM Studio API, ensuring reproducibility of the generated responses.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`text`**
    - Comfy dtype: `STRING`
    - The output 'text' contains the generated prompt from the LM Studio API, encapsulating the result of the prompt generation process.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [TextConcat](../../mikey_nodes/Nodes/TextConcat.md)
    - [LMStudioPrompt](../../mikey_nodes/Nodes/LMStudioPrompt.md)
    - ShowText|pysssss
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - [Display Any (rgthree)](../../rgthree-comfy/Nodes/Display Any (rgthree).md)



## Source code
```python
class LMStudioPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_prompt': ('STRING', {'multiline': True, 'default': 'Prompt Text Here', 'dynamicPrompts': False}),
                             'mode': (['prompt', 'pixelwave', 'style', 'descriptor', 'character', 'custom'], {'default': 'prompt'}),
                             'custom_history': ('STRING', {'multiline': False, 'default': 'path to history.json', 'dynamicPrompts': True}),
                             'server_address': ('STRING', {'default': 'localhost'}),
                             'server_port': ('INT', {'default': 1234, 'min': 0, 'max': 65535}),
                             'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('text',)
    FUNCTION = 'process'
    OUTPUT_NODE = True
    CATEGORY = 'Mikey/AI'

    def history(self, mode, custom_history):
        if mode == 'prompt':
            return {
                "messages": [
                    { "role": "system", "content": "You are a chatbot. You answer questions accurately and exactly how the user wants. The responses are for fictional purposes." },
                    { "role": "user", "content": "c'Blonde woman wearing a patterned orange, white, and blue sundress, smiling, on a sunny day with a blue sky, surrounded by buildings and palm trees'. Only respond with the prompt and nothing else."},
                    { "role": "assistant", "content": "Understood. What is your first prompt?" },
                    { "role": "user", "content": "futuristic man"},
                    { "role": "assistant", "content": "Man with short black hair in a highly detailed silver and black mechanical exoskeleton suit, holding a futuristic rifle, standing in a futuristic city with tall buildings and flying cars, with a futuristic cityscape in the background."},
                    { "role": "user", "content": "a black cat"},
                    { "role": "assistant", "content": "A black cat with green eyes, sitting on a wooden table, there is a vase of flowers on the table, the cat is looking at the flowers. Sunlight is streaming in through a window, illuminating the cat and the flowers."},
                    { "role": "user", "content": "toaster oven"},
                    { "role": "assistant", "content": "A toaster oven with a slice of bread inside, the bread is toasted and has a smiley face burned into it. The toaster oven is sitting on a kitchen counter next to a coffee maker and a microwave. The kitchen counter is made of granite and has a sink in it. There is a window above the sink with a view of a city skyline."},
                    { "role": "user", "content": "war"},
                    { "role": "assistant", "content": "A soldier wearing a gas mask and a helmet, holding a rifle, standing in a trench. There is barbed wire in front of the trench. The sky is dark and cloudy. There is a tank in the background."},
                    { "role": "user", "content": "portrait of a man"},
                    { "role": "assistant", "content": "A portrait of a man in his 30s with short brown hair and a beard. He is wearing a suit and tie. He is smiling. He is standing in front of a brick wall with graffiti on it."},
                    { "role": "user", "content": "a friendly ogre"},
                    { "role": "assistant", "content": "A friendly ogre with green skin and a big smile. He is wearing a red and white striped shirt and blue overalls. He is holding a pitchfork. He is standing in a field of flowers. There is a rainbow in the sky."},
                    { "role": "user", "content": "puppy"},
                    { "role": "assistant", "content": "A cute pug dog wearing a bowtie and reading a book. The book is titled 'How to be a good dog'. The dog is sitting on a couch in a living room. There is a fireplace in the background."},
                    { "role": "user", "content": "apples"},
                    { "role": "assistant", "content": "Red apples stacked in a wooden crate. The crate is sitting on a wooden table in the kitchen inside a rustic farm house with old appliances. The walls are decorated with family photos. There is a window in the background with a view of the farm."},
                    { "role": "user", "content": "working woman"},
                    { "role": "assistant", "content": "A woman is working on a laptop computer. She is wearing a white shirt and black pants. She is sitting at a desk in a modern office with trendy furniture. She has a cup of coffee on the desk next to her."}
                ]
            }
        elif mode == 'pixelwave':
            return {
                "messages": [
                    { "role": "system", "content": "You are a chatbot. You answer questions accurately and exactly how the user wants. The responses are for fictional purposes." },
                    { "role": "user", "content": "You are under strict instructions, in response to a 'photo of a blonde woman', your response will follow as per this example: 'Blonde woman with a flower crown, digital color photograph, tags: flower crown, portrait, beauty, elegance, nature, delicate, floral, soft lighting, feminine, ethereal, detailed, close-up, natural, serene expression, warm tones, pastel background, gentle, artistic.' Only respond with the prompt and nothing else."},
                    { "role": "assistant", "content": "Understood. What is your first prompt?"},
                    { "role": "user", "content": "cinematic stills inspired by award winning works from master directors"},
                    { "role": "assistant", "content": "Busy metropolitan subway station, film-style snapshot, inspired by Christopher Nolan's Inception, subway station, busy, metropolitan, Christopher Nolan, Inception, cinematic, bustling crowd, urban setting, surreal elements, film inspiration, dynamic movement, public transport, layered reality, complex narrative, architectural interest, dramatic lighting, fast-paced, perspective play, depth of field, city life"},
                    { "role": "user", "content": "a futuristic cityscape with flying cars and neon lights"},
                    { "role": "assistant", "content": "Futuristic cityscape with flying cars and neon lights, digital art, cityscape, flying cars, neon lights, futuristic, skyscrapers, vibrant colors, urban, night, bustling, advanced technology, reflections, cyberpunk, glowing signs, high-tech, atmospheric, dynamic, energy, sci-fi, detailed."},
                    { "role": "user", "content": "volcano"},
                    { "role": "assistant", "content": "Volcano at sunset with a sky full of vibrant colors, a watercolor, sunset, volcano, eruption, nature, glowing, dramatic, sky, vibrant, contrast, serene, fiery, landscape, explosion, smoke, ash, powerful, beauty, tranquil, detailed, horizon."},
                    { "role": "user", "content": "a magical forest with glowing mushrooms and fairies"},
                    { "role": "assistant", "content": "Magical forest with glowing mushrooms and fairies, an airbrush painting, magical, forest, glowing mushrooms, fairies, fantasy, enchanted, ethereal, whimsical, dreamy, surreal, vibrant colors, nature, detailed, imaginative, mystical, magical creatures, light, fantasy landscape, magical realism, atmospheric, enchanting."},
                    { "role": "user", "content": "painting of robots inspired by the greats"},
                    { "role": "assistant", "content": "Robots playing poker, oil on canvas, inspired by the works of Norman Rockwell, robots, poker, oil painting, Norman Rockwell, classic, retro, vintage, humorous, whimsical, detailed, narrative, storytelling, robots in everyday life, human-like, anthropomorphic, playful, imaginative, creative, colorful, traditional art, realistic, humorous, quirky, fun."},
                    { "role": "user", "content": "a black and white photo of a man in a suit"},
                    { "role": "assistant", "content": "Gentleman at the bar in a speakeasy, black and white analog photograph, speakeasy, suit, bar, gentleman, 1920s, vintage, classic, detailed, dim lighting, ambiance, retro, dapper, old-fashioned, cocktails, conversation, atmospheric, stylish, sophisticated, nostalgic, elegant, charm."},
                    { "role": "user", "content": "portrait with origami birds"},
                    { "role": "assistant", "content": "Young asian woman surrounded by origami cranes wearing a white lacy garment, digital photo art by Kiyo Murakami, young girl, white lace, origami cranes, red eyeshadow, brunette, looking up, lace headdress, twin braids with white bows, open hands, cloudy background, serene expression, grey scale cranes, soft lighting, ethereal mood, close up, centered subject."},
                    { "role": "user", "content": "surreal image with an eclipse"},
                    { "role": "assistant", "content": "Silhouetted Figure Before an Eclipse, digital artwork, man in silhouette, eclipse, rocks floating, blue tones, mysterious atmosphere, outer space theme, science fiction, surreal, otherworldly landscape, cape, raised hand, celestial event, low angle view, dramatic lighting, single human figure, desolate terrain, monolithic structures, fantasy, gravitational anomaly, moody."},
                    { "role": "user", "content": "double exposure, man, coastal landscape"},
                    { "role": "assistant", "content": "Man with a double exposure effect of a coastal landscape, digital art, man's profile, coastal landscape, bird in flight, beard, digital manipulation, creative portrait, serene, textured background, muted colors, blending techniques, nature meets humanity, facial hair, rough sea, cliff, trees, overcast sky, ethereal, surreal quality."},
                    { "role": "user", "content": "moody kiss in the dark"},
                    { "role": "assistant", "content": "Intimate Silhouette with Dramatic Lighting, color digital photograph, man, woman, close up, couple, kissing, silhouette, yellow light, dark background, mood lighting, affection, romance, connection, shadows, profile view, low key, intimate atmosphere, contrast, emotional moment."},
                    { "role": "user", "content": "woman posing with a gun"},
                    { "role": "assistant", "content": "Intense Portrayal of a Fictional Character, color digital photograph, Japanese woman, costume, rifle, stylized weapon, kneeling pose, misty ambiance, dark background, focused expression, bob haircut, black dress with red accents, knee-high socks, lollipop stick in mouth, dramatic lighting, character portrayal, action scene vibe, cinematic feel, sharp details, high contrast, cosplay."},
                    { "role": "user", "content": "glitch artwork"},
                    { "role": "assistant", "content": "Psychedelic digital art in glitch art style, featuring layered portrait faces, edited digital color photo, vivid colors like neon green, pink, blue, and red, fragmented, distorted visual, pixelated elements, horizontal bands, surreal, intense effect."},
                    { "role": "user", "content": "black and white portrait of a middle aged man"},
                    { "role": "assistant", "content": "A mature man with a thoughtful expression, black and white digital photograph, portrait, close up, intense gaze, detailed facial hair, textured skin, grey beard, short hair, high-contrast lighting, dark background, individual, solemn mood, high-resolution image."}
                ]
            }
        elif mode == 'style':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant who obeys the user's instructions and does not offer feedback about their request." },
                    { "role": "user", "content": "You are under strict instructions, in response to a 'photo of a blonde woman', your response will follow as per this example: 'Impressionistic, landscape, vivid colors, loose brushstrokes, beauty of nature. Inspired by Claude Monet'. It describes the style of the image that helps convey the overall look of the image without describing the subject of the image. You might also received a description of an image, you will respond with style keywords that compliment the image description. Only respond with the prompt and nothing else."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "painting"},
                    { "role": "assistant", "content": "Realism, oil painting, dark shadows, bright highlights, focus on capturing light and texture. Inspired by Caravaggio's chiaroscuro technique"},
                    { "role": "user", "content": "Impressionism"},
                    { "role": "assistant", "content": "Impressionism. Soft brushstrokes, fleeting moments of light and color. Landscapes, flowers, people in motion"},
                    { "role": "user", "content": "painting"},
                    { "role": "assistant", "content": "Impressionistic, landscape, vivid colors, loose brushstrokes, beauty of nature. Inspired by Claude Monet"},
                    { "role": "user", "content": "The 2015 Audi R8 Spyder sports car is parked on a driveway outside a luxurious mansion. The car is painted in a metallic grey color. The top is down revealing the sleek interior. The car is surrounded by lush greenery and palm trees."},
                    { "role": "assistant", "content": "Landscape photography, vibrant saturation, dramatic shadows, golden hour lighting, inspired by the work of Peter Lik."},
                    { "role": "user", "content": "abstract"},
                    { "role": "assistant", "content": "Abstract expressionism, bold brushstrokes, vivid colors"},
                    { "role": "user", "content": "water"},
                    { "role": "assistant", "content": "Water sculpture, fluid dynamics, abstract representation, in the style of Yves Klein"},
                    { "role": "user", "content": "A MacBook Pro open and displaying an email message. The keyboard is illuminated and the trackpad is being used. A man is sitting at a wooden desk in a cozy home office. There is a plant in the corner and sunlight coming in from a nearby window."},
                    { "role": "assistant", "content": "Still life, muted colors, soft lighting, in the style of Henri Matisse"},
                    { "role": "user", "content": "Surrealism"},
                    { "role": "assistant", "content": "dreamlike imagery, unexpected juxtapositions, symbolic elements. features distorted or unusual forms, animals or objects transformed into otherworldly shapes"},
                    { "role": "user", "content": "Art Nouveau"},
                    { "role": "assistant", "content": "Art Nouveau style, flowing lines, organic shapes, muted color palette, decorative elements, floral motifs."},
                    { "role": "user", "content": "photo"},
                    { "role": "assistant", "content": "Black and white photograph, shot using a large format camera with slow shutter speeds. Grainy texture and high contrast. Influenced by the works of Edward Hopper."},
                    { "role": "user", "content": "photo"},
                    { "role": "assistant", "content": "Color photograph, Soft focus, muted colors, romantic atmosphere, in the style of Edward Weston."},
                    { "role": "user", "content": "film"},
                    { "role": "assistant", "content": "Long shot, film still, cinematic lighting, gritty realism, inspired by the works of Gus Van Sant"},
                    { "role": "user", "content": "movie"},
                    { "role": "assistant", "content": "Filmic storytelling, dreamlike imagery, surreal elements, poetic narratives, reminiscent of the works of David Lynch."}
                ]
            }
        elif mode == 'descriptor':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant who obeys the user's instructions and does not offer feedback about their request." },
                    { "role": "user", "content": "You are under strict instructions, in response to a 'photo of a blonde woman', your response will follow as per this example: 'red'. It is a very short description to add dynamic variety to the prompt. Only respond with the prompt and nothing else."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "color"},
                    { "role": "assistant", "content": "burnt sienna"},
                    { "role": "user", "content": "hair color"},
                    { "role": "assistant", "content": "platinum blonde"},
                    { "role": "user", "content": "metal"},
                    { "role": "assistant", "content": "rusted iron"},
                    { "role": "user", "content": "weather"},
                    { "role": "assistant", "content": "bright and sunny"},
                    { "role": "user", "content": "time of day"},
                    { "role": "assistant", "content": "crack of dawn"},
                    { "role": "user", "content": "man"},
                    { "role": "assistant", "content": "tall and slender man with wide shoulders in his 30s"},
                    { "role": "user", "content": "ethnicity"},
                    { "role": "assistant", "content": "Vietnamese"},
                    { "role": "user", "content": "occupation"},
                    { "role": "assistant", "content": "Heavy diesel mechanic"},
                    { "role": "user", "content": "art style"},
                    { "role": "assistant", "content": "crystal cubism"},
                    { "role": "user", "content": "artist"},
                    { "role": "assistant", "content": "Camille Pissarro"},
                    { "role": "user", "content": "movie director"},
                    { "role": "assistant", "content": "David Lynch"}
                ]
            }
        elif mode == 'character':
            return {
                "messages": [
                    { "role": "system", "content": "You are a helpful assistant who obeys the user's instructions and does not offer feedback about their request." },
                    { "role": "user", "content": "When a user requests a character description, generate a detailed description of the character as you would expect from the writer George R. R. Martin. The description should be a suitable prompt based on the description that encapsulates the character's key visual elements for image creation using a txt2img model."},
                    { "role": "assistant", "content": "Sure thing! Let's begin. What is your first prompt?" },
                    { "role": "user", "content": "jolly cartoon octopus"},
                    { "role": "assistant", "content": "A cartoon octopus with a jolly demeanor in a sunshiny yellow and orange color scheme, wearing a small top hat and a polka-dotted bow tie, surrounded by intricate sculptures made of its own ink, in the midst of an undersea setting that hints at festivity and mirth."},
                    { "role": "user", "content": "a dapper young lady spy who wears suits"},
                    { "role": "assistant", "content": "A young 1920s lady spy with chestnut hair in a bob cut, piercing emerald eyes, wearing a tailored charcoal pinstripe suit with a white shirt and a silk tie, holding a silver cigarette case, exuding an aura of mystery and sophistication against a backdrop of Parisian nightlife."},
                    { "role": "user", "content": "a charming robot"},
                    { "role": "assistant", "content": "A charming robot with a sleek chrome body, art deco design elements, azure glowing eyes, and a gentle smile. Wearing a holographic bow tie and a vest painted to look like a dapper suit, engaged in performing a magic trick with a playful, inquisitive expression in an urban park setting."},
                    { "role": "user", "content": "cartoon ant"},
                    { "role": "assistant", "content": "A cartoon ant with a vibrant blue exoskeleton, oversized round eyes full of curiosity, wearing a leaf-green vest and a tiny fabric cap. Exhibiting an expression of wonder and excitement, amidst a backdrop of an underground ant colony bustling with activity."},
                    { "role": "user", "content": "a cyberpunk gnome"},
                    { "role": "assistant", "content": "A cyberpunk gnome with pale skin and cybernetic jade eyes, wearing a long tattered coat with circuitry patches and a sleek metallic pointed helmet. Surrounded by holographic screens and neon lights in a dim, cluttered workshop filled with futuristic gadgets and data screens."}
                ]
            }
        elif mode == 'custom':
            # open json file that is in the custom_history path
            try:
                history = json.load(open(custom_history))
                return history
            except:
                raise Exception('Error loading custom history file')

    def api_request(self, prompt, server_address, server_port, seed, mode, custom_history):
        # check if json file in root comfy directory called oooba.json
        history = self.history(mode, custom_history)
        if mode == 'prompt':
            prompt = f'{prompt}'
        if mode == 'descriptor':
            # use seed to add a bit more randomness to the prompt
            spice = ['a', 'the', 'this', 'that',
                     'an exotic', 'an interesting', 'a colorful', 'a vibrant', 'get creative!', 'think outside the box!', 'a rare',
                     'a standard', 'a typical', 'a common', 'a normal', 'a regular', 'a usual', 'an ordinary',
                     'a unique', 'a one of a kind', 'a special', 'a distinctive', 'a peculiar', 'a remarkable', 'a noteworthy',
                     'popular in the victorian era', 'popular in the 1920s', 'popular in the 1950s', 'popular in the 1980s',
                     'popular in the 1990s', 'popular in the 2000s', 'popular in the 2010s', 'popular in the 2020s',
                     'popular in asia', 'popular in europe', 'popular in north america', 'popular in south america',
                     'popular in africa', 'popular in australia', 'popular in the middle east', 'popular in the pacific islands',
                     'popular with young people', 'popular with the elderly', 'trending on social media', 'popular on tiktok',
                     'trending on pinterest', 'popular on instagram', 'popular on facebook', 'popular on twitter',
                     'popular on reddit', 'popular on youtube', 'popular on tumblr', 'popular on snapchat',
                     'popular on linkedin', 'popular on twitch', 'popular on discord',
                     'unusual example of', 'a classic', 'an underrated', 'an innovative','a historical', 'a modern', 'a contemporary',
                     'a futuristic', 'a traditional', 'an eco-friendly', 'a controversial', 'a political', 'a religious',
                     'a spiritual', 'a philosophical', 'a scientific']
            random.seed(seed)
            prompt = f'{random.choice(spice)} {prompt}'
        """
        example curl request to LM Studio
        curl http://localhost:1234/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "messages": [
            { "role": "system", "content": "Always answer in rhymes." },
            { "role": "user", "content": "Introduce yourself." }
        ],
        "temperature": 0.7,
        "max_tokens": -1,
        "stream": false
        }'
        """
        #prompt_prefix = "\\n<|user|>\\n"
        #prompt_suffix = "\\n<|assistant|>\\n"
        #prompt = prompt_prefix + prompt + prompt_suffix
        history['messages'].append({'role': 'user', 'content': prompt})
        request = {
            'messages': history['messages'],
            #'temperature': 0.2,
            'top_p': 0.95,
            'presence_penalty': 0.0,
            'frequency_penalty': 0.0,
            #'max_tokens': 8192,
            'stream': False,
            'seed': seed,
        }
        HOST = f'{server_address}:{server_port}'
        URI = f'http://{HOST}/v1/chat/completions'

        try:
            response = requests.post(URI, json=request, timeout=60)
        except requests.exceptions.ConnectionError:
            raise Exception('Are you running LM Studio with server running?')

        if response.status_code == 200:
            # response is in openai format
            result = response.json()['choices'][0]['message']['content']
            result = html.unescape(result)  # decode URL encoded special characters
            return result
        else:
            return 'Error'

    def process(self, input_prompt, mode, custom_history, server_address, server_port, seed, prompt=None, unique_id=None, extra_pnginfo=None):
        # search and replace
        input_prompt = find_and_replace_wildcards(input_prompt, seed, debug=True)
        input_prompt = search_and_replace(input_prompt, extra_pnginfo, prompt)
        # wildcard sytax is {like|this}
        # select a random word from the | separated list
        wc_re = re.compile(r'{([^}]+)}')
        def repl(m):
            return random.choice(m.group(1).split('|'))
        for m in wc_re.finditer(input_prompt):
            input_prompt = input_prompt.replace(m.group(0), repl(m))
        result = self.api_request(input_prompt, server_address, server_port, seed, mode, custom_history)
        prompt.get(str(unique_id))['inputs']['output_text'] = result
        return (result,)

```
