import os
import sys
import time
import json
import argparse

from utils import console
from utils.comfy import execute_prompt
from agent.cot import CoTPipeline
from agent.rag import RAGPipeline
from agent.crp import CRPPipeline
from agent.mad import MADPipeline
from agent.cotsc import CoTSCPipeline
from agent.comfy import ComfyPipeline
from agent.fewshot import FewShotPipeline
from agent.zeroshot import ZeroShotPipeline
from agent.variant.comfy_no_adapt import ComfyNoAdaptPipeline
from agent.variant.comfy_no_refine import ComfyNoRefinePipeline
from agent.variant.comfy_no_combine import ComfyNoCombinePipeline
from agent.variant.comfy_no_retrieve import ComfyNoRetrievePipeline
from agent.variant.rag_json_representation import RAGJsonRepresentationPipeline
from agent.variant.rag_list_representation import RAGListRepresentationPipeline


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruction', type=str, required=True)
    parser.add_argument('--agent_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--num_examples', type=int, default=3)
    parser.add_argument('--num_trajectories', type=int, default=3)
    parser.add_argument('--num_references', type=int, default=5)
    parser.add_argument('--num_solvers', type=int, default=3)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--step_limitation', type=int, default=5)
    parser.add_argument('--debug_limitation', type=int, default=1)
    args = parser.parse_args()

    if args.save_path is None:
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        args.save_path = f'./cache/pipeline_record/{timestamp}'
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/instruction.txt', 'w') as file:
        file.write(args.instruction)
    sys.stdout = console.Logger(f'{args.save_path}/logging.txt')

    # create pipeline
    print(' Program Status '.center(80, '-'))
    print('creating pipeline...')
    print()

    if args.agent_name == 'zeroshot':
        pipeline = ZeroShotPipeline()
    elif args.agent_name == 'fewshot':
        pipeline = FewShotPipeline(
            num_examples=args.num_examples
        )
    elif args.agent_name == 'cot':
        pipeline = CoTPipeline(
            num_examples=args.num_examples
        )
    elif args.agent_name == 'cotsc':
        pipeline = CoTSCPipeline(
            num_examples=args.num_examples,
            num_trajectories=args.num_trajectories
        )
    elif args.agent_name == 'rag':
        pipeline = RAGPipeline(
            num_references=args.num_references
        )
    elif args.agent_name == 'crp':
        pipeline = CRPPipeline(
            num_references=args.num_references,
            num_rounds=args.num_rounds
        )
    elif args.agent_name == 'mad':
        pipeline = MADPipeline(
            num_references=args.num_references,
            num_solvers=args.num_solvers,
            num_rounds=args.num_rounds
        )
    elif args.agent_name == 'comfy':
        pipeline = ComfyPipeline(
            num_references=args.num_references,
            step_limitation=args.step_limitation,
            debug_limitation=args.debug_limitation
        )
    elif args.agent_name == 'rag_json_representation':
        pipeline = RAGJsonRepresentationPipeline(
            num_references=args.num_references
        )
    elif args.agent_name == 'rag_list_representation':
        pipeline = RAGListRepresentationPipeline(
            num_references=args.num_references
        )
    elif args.agent_name == 'comfy_no_combine':
        pipeline = ComfyNoCombinePipeline(
            num_references=args.num_references,
            step_limitation=args.step_limitation
        )
    elif args.agent_name == 'comfy_no_adapt':
        pipeline = ComfyNoAdaptPipeline(
            num_references=args.num_references,
            step_limitation=args.step_limitation
        )
    elif args.agent_name == 'comfy_no_retrieve':
        pipeline = ComfyNoRetrievePipeline(
            num_references=args.num_references,
            step_limitation=args.step_limitation
        )
    elif args.agent_name == 'comfy_no_refine':
        pipeline = ComfyNoRefinePipeline(
            num_references=args.num_references,
            step_limitation=args.step_limitation
        )

    # generate workflow
    print(' Program Status '.center(80, '-'))
    print('generating workflow...')
    print()
    try:
        prompt = pipeline(args.instruction)
        with open(f'{args.save_path}/workflow.json', 'w') as file:
            json.dump(prompt, file, indent=4)
    except Exception as error:
        print(' Program Status '.center(80, '-'))
        print(f'failed to generate workflow: {error}')
        print()
        return

    # execute workflow
    print(' Program Status '.center(80, '-'))
    print('executing workflow...')
    print()
    try:
        _, outputs = execute_prompt(prompt)
        for name, data in outputs.items():
            with open(f'{args.save_path}/{name}', 'wb') as file:
                file.write(data)
    except Exception as error:
        print(' Program Status '.center(80, '-'))
        print(f'failed to execute workflow: {error}')
        print()
        return


if __name__ == '__main__':
    main()
