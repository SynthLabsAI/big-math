from datasets import load_dataset
import re

from utils import DATASET_PATH

def add_yes_no_signal(row):
    row['is_yes_no_question'] = False


    # check if the solution responds with yes/no
    # if final answer is not empty, check that first
    if row['final_answer']:
        if re.search(r'(\s|\{|\b)(yes|no)(\s|\}|\b)', row['final_answer'], re.IGNORECASE):
            row['is_yes_no_question'] = True
    else:
        # check the final line of the solution
        solution_lines = row['solution'].split('\n')
        if re.search(r'(\s|\{|\b)(yes|no)(\s|\}|\b)', solution_lines[-1], re.IGNORECASE):
            row['is_yes_no_question'] = True

    # fall back to checking if the problem sounds like a yes/no question
    problem_lines = row['problem'].split('\n')

    # search for a series of yes/no question prefixes
    yes_no_question_prefixes = [
        "is ", "are ", "do ", "does ", "can "
    ]
    # check if there is a previous line, if so then the previous line must end with punctuation ".,:;!?".
    if (len(problem_lines) > 1 and re.search(r'[.,:;!?\s]$', problem_lines[-2])) or \
        len(problem_lines) == 1:
        if any(problem_lines[-1].lower().startswith(prefix) for prefix in yes_no_question_prefixes):
            row['is_yes_no_question'] = True

    return row

def main():
    # load the dataset
    dataset = load_dataset(DATASET_PATH, split="train")

    # run the detection over the full dataset
    dataset = dataset.map(add_yes_no_signal)

    # push the dataset
    dataset.push_to_hub(DATASET_PATH)

if __name__ == '__main__':
    main()