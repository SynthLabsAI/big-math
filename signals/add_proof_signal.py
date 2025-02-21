from datasets import load_dataset
import multiprocessing as mp
import nltk

nltk.download('punkt_tab')

# load the utilities to add a new signal to the master dataset
from utils import DATASET_PATH

# given a single row of the dataset, determine if there is a proof
def has_proof(row):
    row['is_math_proof'] = False

    lower_problem = row['problem'].lower()
    if 'prove that' in lower_problem or 'a proof' in lower_problem:
        row['is_math_proof'] = True

    # special search in "olympiads" subset, search for "show" at the beginning of a sentence
    if row['source'] == "olympiads" and not row['is_math_proof']:
        sentences = nltk.sent_tokenize(row['problem'])
        for sentence in sentences:
            sentence = sentence.lstrip().lower()
            if sentence.startswith('show'):
                row['is_math_proof'] = True
                break

    return row

def main():

    # load the dataset
    dataset = load_dataset(DATASET_PATH, split="train")

    # run proof detection over the full dataset
    dataset = dataset.map(has_proof, num_proc=mp.cpu_count())

    # push the dataset
    dataset.push_to_hub(DATASET_PATH)

if __name__ == "__main__":
    main()