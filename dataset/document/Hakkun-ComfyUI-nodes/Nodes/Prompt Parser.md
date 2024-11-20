---
tags:
- Prompt
---

# Prompt Parser
## Documentation
- Class name: `Prompt Parser`
- Category: `Hakkun`
- Output node: `False`

The Prompt Parser node is designed to process and transform text prompts based on a set of rules and conditions. It can parse input prompts to extract and format specific pieces of information, apply random selection to parts of the text based on predefined conditions, and integrate external tags into the prompt. This node is essential for dynamically generating or modifying text prompts in a controlled manner, allowing for customized and context-aware text output.
## Input types
### Required
- **`prompt`**
    - The main text input that the node will process. It serves as the base content for parsing and applying transformations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`tags_file`**
    - A file path to a text file containing tags that can be used to replace placeholders within the prompt. This allows for dynamic content insertion based on external data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the random selections made within the prompt processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`extra1`**
    - An optional text input for additional content that can be inserted into the prompt at a specified placeholder.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`extra2`**
    - Another optional text input for extra content, similar to 'extra1', providing further customization capability for the prompt.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`tags`**
    - Directly provides tags as text input for replacing placeholders in the prompt, offering an alternative to using a tags file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`positive`**
    - Comfy dtype: `STRING`
    - The processed text output containing elements designated as positive or desirable.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `STRING`
    - The processed text output containing elements designated as negative or undesirable.
    - Python dtype: `str`
- **`debug`**
    - Comfy dtype: `STRING`
    - A detailed log of the parsing process, including seed values, extra inputs, and the raw prompt before processing, useful for debugging and analysis.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class PromptParser:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (TEXT_TYPE, {"default": '', "multiline": True}),
                "tags_file": ("STRING", {"default": '', "multiline": False}),
                "seed": (INT_TYPE, {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "extra1": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "extra2": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "tags": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,TEXT_TYPE,TEXT_TYPE)
    RETURN_NAMES = ("positive","negative","debug")
    FUNCTION = "parse_prompt"

    CATEGORY = "Hakkun"

    def format_commas(self, input):
        formatted_string = input.replace(' ,', ',').replace(',', ', ')
        return formatted_string

    def line_perc(self, input):
        pattern = r'(\d+)%'
        match = re.search(pattern, input)
        if match:
            percentage = int(match.group(1)) / 100.0
            return percentage
        else:
            return 0.5

    def get_perc_text(self, input):
        pattern = r'(\d+)%(.*)'
        match = re.search(pattern, input)
        if match:
            rest_of_string = match.group(2)
            return rest_of_string.strip()
        else:
            return input

    def parse(self, text,  delimiter=","):
        result = []
        positives = []
        negatives = []

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("^"):
                break
            if line.startswith("!"):
                continue
            if self.is_empty_or_whitespace(line):
                continue

            if line.startswith("?"):
                line = line[1:]
                perc = self.line_perc(line)
                if random.random() > perc:
                    continue
                line = self.get_perc_text(line)

            if "@" in line:
                positive, negative = line.split("@")
                positives.append(positive)
                negatives.append(negative)
            else:
                positives.append(line)

        positives = self.remove_empty(positives)
        negatives = self.remove_empty(negatives)

        if positives:
            result.append(delimiter.join(positives))
        else:
            result.append("")
        if negatives:
            result.append(delimiter.join(negatives))
        else:
            result.append("")

        return result

    def is_empty_or_whitespace(self, input):
        return input.strip() == ''

    def replace_multiple_spaces(self, input):
        return re.sub(r'\s+', ' ', input.strip())

    def replace_random(self, match):
        options = match.group(1).split('*')
        return random.choice(options)

    def select_random_from_braces(self, input):
        pattern = r'\[(.*?)\]'
        result = re.sub(pattern, self.replace_random, input)
        return result

    def randomly_select_string_with_weight(self, arr, n):
        weights = []        
        strings_without_weight = []
        weight_pattern = r'(\*(\d+)\*|\d+:)'

        for string in arr:
            match = re.search(weight_pattern, string)
            if match:
                weight_str = match.group(1)

                # Extract "*weight*" or "weight:" patterns
                if ':' in weight_str:
                    weight = int(weight_str[:-1])
                else:
                    weight = int(weight_str.strip('*'))
                weights.append(weight)
                strings_without_weight.append(
                    string.replace(match.group(), ''))
            else:
                # If no weight pattern is found, consider weight as 1
                weights.append(100)
                strings_without_weight.append(string)
        
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        # Select no more than the number of strings available
        if n > len(strings_without_weight):
            n = len(strings_without_weight)
        
        selected_string = np.random.choice(strings_without_weight, n, p=normalized_weights, replace=False)
        return selected_string

    # "I went there with [a [fast|slow] [car|[boat|yaht]]|an [expensive|cheap] [car|[boat|yaht]]]"
    # [[*10*pink|blue] bedroom*100*|city at [day|night] with [cars|trains|rockets]]
    # [*150*car|boat*30*|bi*80*ke]
    def select_random(self, text):
        def random_choice(match):
            
            num_choices, options_str = match.groups()
            n = 1
            if num_choices:
                # Randomly select between n-m choices
                if '-' in num_choices:
                    start, end = map(int, num_choices.split('-'))
                    n = random.randint(start, end)
                # Randomly select n choices
                else:
                    n = int(num_choices)
            
            options = options_str.split('|')
            return ' '.join(self.randomly_select_string_with_weight(options, n))

        pattern = r'\[(?:(\d+-\d+|\d*)#)?([^\[\]]+)\]'

        while re.search(pattern, text):
            text = re.sub(pattern, random_choice, text)

        return text

    def remove_empty(self, arr):
        return [s for s in arr if s.strip()]

    def process_extra(self, text, placeholder, extra=None):
        if isEmpty(extra):
            if placeholder in text:
                return text.replace(placeholder, '')
            return text
        if placeholder in text:
            return text.replace(placeholder, extra)
        return extra +', '+ text

    def fix_commas(self, text):
        elements = text.split(",")
        elements = [element.strip() for element in elements]
        elements = [element for element in elements if element]
        return ", ".join(elements)

    def parse_prompt(self, prompt, tags_file, seed, extra1=None, extra2=None, tags=None):
        random.seed(seed)
        np_seed = seed % 2**32
        np.random.seed(np_seed)
        if isOk(tags_file):
            tags = load_text(tags_file)

        prompt = remove_comments(prompt)

        prompt = self.process_extra(prompt, "<extra2>", extra2)
        prompt = self.process_extra(prompt, "<extra1>", extra1)

        if isOk(tags):
            tags = remove_empty_lines(tags)
            tags_dict = multiline_string_to_dict(tags)
            prompt = replace_placeholders(prompt, tags_dict)

        raw = prompt

        prompt = prompt.replace("em:", "embedding:")

        prompt = self.select_random(prompt)

        result = self.parse(prompt)

        result[0] = self.fix_commas(result[0])
        result[1] = self.fix_commas(result[1])

        return  (
            result[0],
            result[1],
            'POSITIVE:\n' + result[0] +
            '\n\nNEGATIVE:\n' + result[1] +
            '\n\nseed:' + str(seed) +
            '\n\nextra1:' + (extra1 or "<none>") +
            '\nextra2:' + (extra2 or "<none>") +
            '\n\nRAW:\n' + raw
        )

```
