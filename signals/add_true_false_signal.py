from datasets import load_dataset
import re
import multiprocessing as mp

from utils import DATASET_PATH

def is_true_false(row):
    row['is_true_false_question'] = False

    # check if solution has flexible "true or false" in it

    # if answer is not empty, check that first:
    if row['final_answer']:
        if re.search(r'(true|false)', row['final_answer'], re.IGNORECASE):
            row['is_true_false_question'] = True
    else:
        # check the final line of the solutions
        for solution in row['hard_math_solutions']:
            lines = solution.split('\n')
            if re.search(r'(true|false)', lines[-1], re.IGNORECASE):
                row['is_true_false_question'] = True

    return row

def main():

    # load the dataset
    dataset = load_dataset(DATASET_PATH, split="train")

    # run proof detection over the full dataset
    dataset = dataset.map(is_true_false, num_proc=mp.cpu_count())

    # push the dataset
    dataset.push_to_hub(DATASET_PATH)

if __name__ == "__main__":
    main()