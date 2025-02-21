from datasets import load_dataset
import multiprocessing as mp

# load the utilities to add a new signal to the master dataset
from utils import DATASET_PATH

LETTER_OPTIONS=['A', 'B', 'C', 'D']
NUMERICAL_OPTIONS=['1', '2', '3', '4']

def is_multiple_choice(question, answer_options):
    # first, filter out a question that uses the answer options as part of the question
    # for example ABCD can be part of the question, but is not an answer option
    question = question.replace("".join(answer_options), "")

    # next, search for the last occurrence of each answer option
    options_found = [question.rfind(option) for option in answer_options[::-1]]

    # if any answer option is not found, return False
    if any(option == -1 for option in options_found):
        return False

    # check if the options are found in reverse order
    for i, option in enumerate(options_found):
        if i > 0 and option > options_found[i - 1]:
            return False

    return True

def detect_multiple_choice(row):
    if is_multiple_choice(row['problem'], LETTER_OPTIONS) or is_multiple_choice(row['problem'], NUMERICAL_OPTIONS):
        row['is_multiple_choice'] = True
    else:
        row['is_multiple_choice'] = False
    return row

def main():

    # pull the dataset
    dataset = load_dataset(DATASET_PATH, split="train")

    # run language detection over the full dataset
    dataset = dataset.map(detect_multiple_choice, num_proc=mp.cpu_count())

    # push the updated dataset
    dataset.push_to_hub(DATASET_PATH)

if __name__ == "__main__":
    main()