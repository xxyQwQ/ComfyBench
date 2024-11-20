import os
import glob
import json
import argparse

from utils import console
from utils.parser import extract_document_with_template


def main():
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--document_folder', type=str, required=True)
    parser.add_argument('--cache_path', type=str, default=None)
    args = parser.parse_args()

    if args.cache_path is None:
        args.cache_path = f'./cache/node_document'
    os.makedirs(f'{args.cache_path}', exist_ok=True)

    # extract document
    meta = {}

    for filepath in glob.glob(f'{args.document_folder}/*/Nodes/*.md'):
        filename, _ = os.path.splitext(os.path.basename(filepath))
        if filename in meta:
            print(console.red(f'already existed: {filename}'))
            continue
        print(console.green(f'creating document: {filename}'))

        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        try:
            document, archive = extract_document_with_template(filename, content)
            meta[filename] = archive
        except Exception as error:
            print(console.red(f'invalid document: {filename}'))
            continue
        with open(f'{args.cache_path}/{filename}.md', 'w', encoding='utf-8') as file:
            file.write(document)

    with open(f'{args.cache_path}/meta.json', 'w', encoding='utf-8') as file:
        json.dump(meta, file, indent=4)


if __name__ == '__main__':
    main()
