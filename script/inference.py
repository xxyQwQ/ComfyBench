import os
import sys
import time
import json
import argparse
import multiprocessing as mp

from utils import console
from agent.cot import CoTPipeline
from agent.rag import RAGPipeline
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


def process(arguments: dict) -> dict:
    print('working on benchmark task: ', arguments['index'], arguments['task']['name'])
    returns = {'index': arguments['index']}
    try:
        sys.stdout = console.Logger(f'{arguments["root"]}/logging/{arguments["index"]}.txt')
        prompt = arguments['pipeline'](arguments['task']['instruction'])
        with open(f'{arguments["root"]}/workflow/{arguments["index"]}.json', 'w') as file:
            json.dump(prompt, file, indent=4)
    except Exception as error:
        print(f'failed to generate workflow: {error}')
    finally:
        sys.stdout = sys.__stdout__
        return returns

def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, required=True)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_examples', type=int, default=3)
    parser.add_argument('--num_trajectories', type=int, default=3)
    parser.add_argument('--num_references', type=int, default=5)
    parser.add_argument('--step_limitation', type=int, default=5)
    parser.add_argument('--debug_limitation', type=int, default=1)
    args = parser.parse_args()

    if args.save_path is None:
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        args.save_path = f'./cache/inference/{timestamp}'
    os.makedirs(f'{args.save_path}/logging', exist_ok=True)
    os.makedirs(f'{args.save_path}/workflow', exist_ok=True)

    # create pipeline
    print(f'creating pipeline for agent: {args.agent_name}')
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
    with open(f'./dataset/benchmark/instruction/complete.json', 'r') as file:
        benchmark = json.load(file)

    pool = mp.Pool(processes=args.num_workers)
    results = []

    for index, task in benchmark.items():
        arguments = {
            'root': args.save_path,
            'pipeline': pipeline,
            'index': index,
            'task': task
        }
        results.append(pool.apply_async(process, args=[arguments]))

    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
