import json

from bs4 import BeautifulSoup

from utils.model import invoke_completion
from utils.parser import parse_code_to_prompt


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


generator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

## Format

You should provide your Python code to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your code with the required format.
'''


def format_generator_agent_prompt(instruction: str) -> str:
    instruction_content = instruction
    prompt_content = generator_prompt.format(
        instruction=instruction_content.strip()
    )
    return prompt_content


def parse_generator_agent_answer(answer: str) -> str:
    soup = BeautifulSoup(answer, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    return code


class ZeroShotPipeline(object):
    def __init__(self):
        pass

    def __call__(self, instruction: str) -> dict:
        # start pipeline
        print(' Task Instruction '.center(80, '-'))
        print(instruction)
        print()

        # generate workflow
        message = format_generator_agent_prompt(instruction)
        print(' Generator Prompt '.center(80, '-'))
        print(message)
        print()
        generation, usage = invoke_completion(message)
        print(' Generator Answer '.center(80, '-'))
        print(generation)
        print(usage)
        print()

        # parse answer
        code = parse_generator_agent_answer(generation)
        print(' Parsed Code '.center(80, '-'))
        print(code)
        print()

        # parse code
        try:
            prompt = parse_code_to_prompt(code)
        except Exception as error:
            raise RuntimeError('failed to parse code') from error
        print(' Parsed Workflow '.center(80, '-'))
        print(json.dumps(prompt, indent=4))
        print()

        return prompt
