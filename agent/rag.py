import json

from bs4 import BeautifulSoup

from utils.parser import parse_code_to_prompt
from utils.model import ReferenceStorage, invoke_completion


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


analyzer_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

Based on the instruction, analyze the core requirements (e.g. object, style, resolution, etc.) and the expected paradigm (e.g., text-to-image, image-to-image, image-to-video, etc.), so that we can retrieve relevant workflows for your reference. Note that you do not need to provide the workflow. Make sure your analysis is clear and concise within a single paragraph.
'''


def format_analyzer_agent_prompt(instruction: str) -> str:
    instruction_content = instruction
    prompt_content = analyzer_prompt.format(
        instruction=instruction_content.strip()
    )
    return prompt_content


generator_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks. ComfyUI workflows can be formulated into the equivalent Python code, where each statement represents the execution of a single node. You are an expert in ComfyUI, helping users to design workflows according to their requirements.

Now you are required to create a ComfyUI workflow to finish the following task:

{instruction}

The core requirements and the expected paradigm are analyzed as follows:

{analysis}

## Reference

According to the requirements, we have retrieved some relevant workflows which may be helpful:

{reference}

## Format

First, you should provide a plan, including which workflows you will refer to and how you will modify and compose them. Your plan should be enclosed using "<plan>" tag. For example: <plan>I will create my workflow based on "text-to-image" and "image_super_resolution". I will cascade them and rewrite the prompt text.</plan>.

After that, you should provide your Python code as in the reference to formulate the workflow. Each line of code should correspond to a single node, so you should avoid nested calls in a single statement. Your code should be enclosed using "<code>" tag. For example: <code>output = node(input)</code>.

Now, provide your plan and code with the required format.
'''


def format_generator_agent_prompt(instruction: str, analysis: str, reference_list: list) -> str:
    instruction_content = instruction
    analysis_content = analysis

    reference_content = ''
    for reference in reference_list:
        reference_content += f'- Workflow: {reference.metadata["name"]}\n\n'
        reference_content += f'<code>\n{reference.metadata["code"]}\n</code>\n\n'
        reference_content += f'<function>\n{reference.metadata["function"]}\n</function>\n\n'
        reference_content += f'<principle>\n{reference.metadata["principle"]}\n</principle>\n\n'

    prompt_content = generator_prompt.format(
        instruction=instruction_content.strip(),
        analysis=analysis_content.strip(),
        reference=reference_content.strip()
    )
    return prompt_content


def parse_generator_agent_answer(answer: str) -> tuple[str, str]:
    soup = BeautifulSoup(answer, 'html.parser')
    plan = safe_extract_from_soup(soup, 'plan')
    code = safe_extract_from_soup(soup, 'code')
    return plan, code


class RAGPipeline(object):
    def __init__(self, num_references: int = 5):
        self.num_references = num_references

    def __call__(self, instruction: str) -> dict:
        # start pipeline
        print(' Task Instruction '.center(80, '-'))
        print(instruction)
        print()

        # analyze instruction
        message = format_analyzer_agent_prompt(instruction)
        print(' Analyzer Prompt '.center(80, '-'))
        print(message)
        print()
        analysis, usage = invoke_completion(message)
        print(' Analyzer Answer '.center(80, '-'))
        print(analysis)
        print(usage)
        print()

        # retrieve reference
        storage = ReferenceStorage()
        query = f'Instruction: {instruction}\n\nAnalysis: {analysis}'
        reference_list = storage.retrieve(query, count=self.num_references)
        print(' Retrieved Reference '.center(80, '-'))
        for reference in reference_list:
            print(reference.page_content)
            print()

        # generate workflow
        message = format_generator_agent_prompt(instruction, analysis, reference_list)
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
