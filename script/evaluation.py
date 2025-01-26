import io
import os
import sys
import time
import json
import base64
import argparse
import multiprocessing as mp
from copy import deepcopy

import cv2
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup

from utils import console
from utils.model import invoke_vision
from utils.comfy import execute_prompt


t2i_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a cat in an astronaut suit, which is consistent with the instruction. The wall is white, which is different from the "green wall" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


i2i_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-image generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The second image is the generation result, with an actual resolution of {result_resolution}.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result keeps the structure of the input reference, but the car is not removed, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


t2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a text-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The given {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction. Enclose your analysis in the <analysis> tag. For example: <analysis>There is a walking robot, which is consistent with the instruction. However, the scene is a street, which is different from the "forest" in the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


i2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of an image-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first image is the input reference, with an actual resolution of {reference_resolution}. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result contains a moving car, which is consistent with the instruction. However, it fails to follow the style of the input reference.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


v2v_prompt = '''
You are an expert in image and video generation, familiar with the latest tasks and techniques. You are capable of understanding the task instruction, analyzing the generation result, and providing an accurate evaluation. Now you are evaluating the result of a video-to-video generation task. You should be tolerant to the quality of the generation result, and focus on the consistency with the instruction.

The task instruction is described as: {instruction}

The first {reference_frame_count} images are the frames sampled from the input reference, with an actual resolution of {reference_resolution}, duration of {reference_duration} seconds and {reference_frame_rate} frames per second. The remaining {result_frame_count} images are the frames sampled from the generation result, with an actual resolution of {result_resolution}, duration of {result_duration} seconds and {result_frame_rate} frames per second.

First, analyze whether the generation result meets each key point in the instruction based on the input reference. Enclose your analysis in the <analysis> tag. For example: <analysis>The generation result improves the resolution of the input reference. However, it fails to convert the input inference into an oil painting style, which is not consistent with the instruction.</analysis>.

Then, provide a final judgment of whether the generation result complies with the instruction. The judgment should either be "True" or "False". Enclose your judgment in the <judgment> tag. For example: <judgment>False</judgment>.
'''


def encode_image(image: Image.Image, size: tuple[int, int] = (512, 512)) -> str:
    image.thumbnail(size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return base64_image


def load_image(image_path: str, size_limit: tuple[int, int] = (512, 512)) -> tuple[str, dict]:
    meta_info = {}
    image = Image.open(image_path)
    meta_info['width'], meta_info['height'] = image.size
    base64_image = encode_image(image, size_limit)
    return base64_image, meta_info


def load_video(video_path: str, size_limit: tuple[int, int] = (512, 512), frame_limit: int = 5) -> tuple[list, dict]:
    base64_frames = []
    meta_info = {}
    video = cv2.VideoCapture(video_path)
    meta_info['width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta_info['height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta_info['num_frames'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_info['frame_rate'] = int(video.get(cv2.CAP_PROP_FPS))
    meta_info['duration'] = meta_info['num_frames'] / meta_info['frame_rate']

    count = 0
    sample_interval = max(6, meta_info['num_frames'] // frame_limit)
    while video.isOpened():
        status, frame = video.read()
        if not status:
            break
        if count % sample_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            base64_frame = encode_image(image, size_limit)
            base64_frames.append(base64_frame)
        count += 1
    video.release()
    return base64_frames, meta_info


def safe_extract_from_soup(soup: BeautifulSoup, tag: str) -> str:
    element = soup.find(tag)
    if element is None:
        return ''
    return element.text.strip()


def parse_evaluation(evaluation: str) -> tuple[str, str]:
    soup = BeautifulSoup(evaluation, 'html.parser')
    analysis = safe_extract_from_soup(soup, 'analysis')
    judgment = safe_extract_from_soup(soup, 'judgment')
    return analysis, judgment


def evaluate_t2i(task: dict) -> bool:
    result_base64_image, result_meta_info = load_image(task['result'])

    prompt = t2i_prompt.format(
        instruction=task['instruction'],
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_i2i(task: dict) -> bool:
    reference_base64_image, reference_meta_info = load_image(f'./dataset/benchmark/resource/{task["resource"]}')
    result_base64_image, result_meta_info = load_image(task['result'])

    prompt = i2i_prompt.format(
        instruction=task['instruction'],
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{result_base64_image}"
        }
    })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_t2v(task: dict) -> bool:
    result_base64_frames, result_meta_info = load_video(task['result'])

    prompt = t2v_prompt.format(
        instruction=task['instruction'],
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
        result_duration=f'{result_meta_info["duration"]:.2f}',
        result_frame_rate=f'{result_meta_info["frame_rate"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_i2v(task: dict) -> bool:
    reference_base64_image, reference_meta_info = load_image(f'./dataset/benchmark/resource/{task["resource"]}')
    result_base64_frames, result_meta_info = load_video(task['result'])

    prompt = i2v_prompt.format(
        instruction=task['instruction'],
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
        result_duration=f'{result_meta_info["duration"]:.2f}',
        result_frame_rate=f'{result_meta_info["frame_rate"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_base64_image}"
        }
    })
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def evaluate_v2v(task: dict) -> bool:
    reference_base64_frames, reference_meta_info = load_video(f'./dataset/benchmark/resource/{task["resource"]}')
    result_base64_frames, result_meta_info = load_video(task['result'])

    prompt = v2v_prompt.format(
        instruction=task['instruction'],
        reference_frame_count=len(reference_base64_frames),
        reference_resolution=f'{reference_meta_info["width"]}x{reference_meta_info["height"]}',
        reference_duration=f'{reference_meta_info["duration"]:.2f}',
        reference_frame_rate=f'{reference_meta_info["frame_rate"]}',
        result_frame_count=len(result_base64_frames),
        result_resolution=f'{result_meta_info["width"]}x{result_meta_info["height"]}',
        result_duration=f'{result_meta_info["duration"]:.2f}',
        result_frame_rate=f'{result_meta_info["frame_rate"]}'
    )
    print(' Evaluator Prompt '.center(80, '-'))
    print(prompt)
    print()

    content = [{
        'type': 'text',
        'text': prompt
    }]
    for reference_base64_frame in reference_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{reference_base64_frame}"
            }
        })
    for result_base64_frame in result_base64_frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{result_base64_frame}"
            }
        })
    message = [{
        "role": "user",
        "content": content
    }]
    evaluation, usage = invoke_vision(message)
    print(' Evaluator Answer '.center(80, '-'))
    print(evaluation)
    print(usage)
    print()

    analysis, judgment = parse_evaluation(evaluation)
    print(' Parsed Analysis '.center(80, '-'))
    print(analysis)
    print()
    print(' Parsed Judgment '.center(80, '-'))
    print(judgment)
    print()

    return analysis, judgment


def process(arguments: dict) -> dict:
    print(f'evaluating on benchmark task: {arguments["index"]} {arguments["task"]["name"]}')
    returns = {'index': arguments['index'], 'resolved': False}
    try:
        if arguments['task']['result'] is None:
            raise RuntimeError(f'invalid result')
        elif not os.path.exists(arguments['task']['result']):
            raise RuntimeError(f'invalid path')
        if arguments['task']['modality'] == 't2i':
            _, judgment = evaluate_t2i(arguments['task'])
        elif arguments['task']['modality'] == 'i2i':
            _, judgment = evaluate_i2i(arguments['task'])
        elif arguments['task']['modality'] == 't2v':
            _, judgment = evaluate_t2v(arguments['task'])
        elif arguments['task']['modality'] == 'i2v':
            _, judgment = evaluate_i2v(arguments['task'])
        elif arguments['task']['modality'] == 'v2v':
            _, judgment = evaluate_v2v(arguments['task'])
        else:
            raise RuntimeError(f'invalid modality')
        if judgment.strip() == 'True':
            returns['resolved'] = True
    except Exception as error:
        print(f'failed to evaluate generation: {error}')
    finally:
        return returns


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_folder', type=str, required=True)
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--resume_last', action='store_true')
    args = parser.parse_args()

    if args.cache_path is None:
        timestamp = time.strftime('%Y-%m-%d-%H-%M-%S')
        args.cache_path = f'./cache/evaluation/{timestamp}'
    os.makedirs(f'{args.cache_path}', exist_ok=True)
    sys.stdout = console.Logger(f'{args.cache_path}/logging.txt')

    # execute workflow
    with open(f'./dataset/benchmark/instruction/complete.json', 'r') as file:
        benchmark = json.load(file)
    record = deepcopy(benchmark)

    if args.resume_last:
        exists = {}
        for name in os.listdir(args.cache_path):
            if name in ['logging.txt', 'summary.txt', 'result.json']:
                continue
            index = name.split('.')[0]
            exists[index] = name
        last = max(exists.keys())

    for index, task in record.items():
        print(f'executing on benchmark task: {index} {task["name"]}')
        record[index]['result'] = None
        record[index]['passed'] = False

        if args.resume_last and index <= last:
            if index in exists:
                record[index]['result'] = f'{args.cache_path}/{exists[index]}'
                record[index]['passed'] = True
            continue

        try:
            with open(f'{args.submit_folder}/{index}.json', 'r') as file:
                prompt = json.load(file)
            status, outputs = execute_prompt(prompt)
            if status['status_str'] == 'success':
                record[index]['passed'] = True
            else:
                raise RuntimeError(f'invalid status')

            for name, data in outputs.items():
                suffix = name.split('.')[-1]
                if task['modality'].endswith('i') and suffix in ['jpg', 'png']:
                    record[index]['result'] = f'{args.cache_path}/{index}.{suffix}'
                    with open(record[index]['result'], 'wb') as file:
                        file.write(data)
                    break
                elif task['modality'].endswith('v') and suffix in ['gif', 'mp4']:
                    record[index]['result'] = f'{args.cache_path}/{index}.{suffix}'
                    with open(record[index]['result'], 'wb') as file:
                        file.write(data)
                    break

        except Exception as error:
            print(f'failed to execute workflow: {error}')
            continue

    # evaluate result
    pool = mp.Pool(processes=args.num_workers)
    results = []

    for index, task in record.items():
        arguments = {
            'record': record,
            'index': index,
            'task': task
        }
        results.append(pool.apply_async(process, args=[arguments]))

    pool.close()
    pool.join()

    for result in results:
        returns = result.get()
        record[returns['index']]['resolved'] = returns['resolved']

    # summarize result
    with open(f'{args.cache_path}/result.json', 'w') as file:
        json.dump(record, file, indent=4)

    summary = {
        'vanilla': {
            'total': 0,
            'passed': 0,
            'resolved': 0
        },
        'complex': {
            'total': 0,
            'passed': 0,
            'resolved': 0
        },
        'creative': {
            'total': 0,
            'passed': 0,
            'resolved': 0
        }
    }

    for task in record.values():
        summary[task['category']]['total'] += 1
        if task['passed']:
            summary[task['category']]['passed'] += 1
        if task['resolved']:
            summary[task['category']]['resolved'] += 1

    summary = pd.DataFrame(summary).transpose()
    with open(f'{args.cache_path}/summary.txt', 'w') as file:
        file.write(summary.to_string())


if __name__ == '__main__':
    main()
