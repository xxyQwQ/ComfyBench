import json

from bs4 import BeautifulSoup

from utils.parser import parse_code_to_prompt
from utils.model import ReferenceStorage, invoke_completion


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

## Example

Here are some example workflows for your reference:

{example}

## Format

First, you should provide a plan, including which workflows you will refer to and how you will modify and compose them. Your plan should be enclosed using "<plan>" tag. For example: <plan>I will create my workflow based on "text-to-image" and "image_super_resolution". I will cascade them and rewrite the prompt text.</plan>.

After that, you should provide your Python code as in the example to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your plan and code with the required format.
'''


def format_generator_agent_prompt(instruction: str, example_list: list) -> str:
    instruction_content = instruction

    example_content = ''
    for example in example_list:
        example_content += f'- Workflow: {example.metadata["name"]}\n\n'
        example_content += f'<code>\n{example.metadata["code"]}\n</code>\n\n'
        example_content += f'<function>\n{example.metadata["function"]}\n</function>\n\n'
        example_content += f'<principle>\n{example.metadata["principle"]}\n</principle>\n\n'

    prompt_content = generator_prompt.format(
        instruction=instruction_content.strip(),
        example=example_content.strip()
    )
    return prompt_content


def parse_generator_agent_answer(answer: str) -> tuple[str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    plan = safe_extract_from_soup(soup, 'plan')
    code = safe_extract_from_soup(soup, 'code')
    return plan, code


class CoTPipeline(object):
    def __init__(self, num_examples: int = 3):
        self.num_examples = num_examples

    def __call__(self, instruction: str) -> dict:
        # start pipeline
        print(' Task Instruction '.center(80, '-'))
        print(instruction)
        print()

        # create example
        storage = ReferenceStorage()
        example_list = storage.document_list[:self.num_examples]
        print(' Created Example '.center(80, '-'))
        for example in example_list:
            print(example.page_content)
            print()

        # generate workflow
        message = format_generator_agent_prompt(instruction, example_list)
        print(' Generator Prompt '.center(80, '-'))
        print(message)
        print()
        generation, usage = invoke_completion(message)
        print(' Generator Answer '.center(80, '-'))
        print(generation)
        print(usage)
        print()

        # parse answer
        plan, code = parse_generator_agent_answer(generation)
        print(' Parsed Plan '.center(80, '-'))
        print(plan)
        print()
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
