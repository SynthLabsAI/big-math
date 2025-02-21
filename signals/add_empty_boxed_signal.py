from datasets import load_dataset
from functools import partial
import multiprocessing as mp
import re
import argparse

def is_boxed_empty(row, patterns):
    row['is_boxed_empty'] = False
    for pattern in patterns:
        if re.search(pattern, row['solution']):
            row['is_boxed_empty'] = True
            break
    return row

def main(dataset_path):
    # load the dataset
    dataset = load_dataset(dataset_path, split="train")

    # define the regex pattern for an empty boxed solution
    empty_boxed_patterns = [
        r'boxed\{\s*\}',
        r'boxed\{[\s\n\r]*\}',
        r'\\boxed\{\s*\}'
    ]

    # run the detection over the full dataset
    is_boxed_empty_partial = partial(is_boxed_empty, patterns=empty_boxed_patterns)
    dataset = dataset.map(is_boxed_empty_partial, num_proc=mp.cpu_count())

    # add the new column, 'is_boxed_empty', to the dataset
    dataset.push_to_hub(dataset_path)

    # print stats
    boxed_empty_only = dataset.filter(lambda x: x['is_boxed_empty'])
    print(f"Boxed empty: {len(boxed_empty_only)} / {len(dataset)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect empty boxed solutions in a dataset.")
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    main(args.dataset_path)