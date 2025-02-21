from datasets import load_dataset
from functools import partial
import multiprocessing as mp

from utils import DATASET_PATH
from semdedup import semantic_deduplication


def merge_problem_and_answer(row):
    row['problem_answer'] = f"Problem: {row['problem']} Answer: {row['final_answer']}"
    return row

def is_semdedup_duplicate(row, idx, indices_to_remove, epsilon):
    row[f'is_semdedup_duplicate_eps{epsilon}'] = idx in indices_to_remove
    return row


def main():
    EPSILONS = [0.5]


    dataset = load_dataset(DATASET_PATH, split="train")
    print(f"Original Dataset: {dataset}")
    
    # create a new column that merges the problem and the answer
    dataset = dataset.map(merge_problem_and_answer, num_proc=mp.cpu_count())

    # convert to pandas and run semdedup
    df = dataset.data.to_pandas()
    for epsilon in EPSILONS:
        print(f"Running semdedup with epsilon {epsilon}")
        df, indices_to_remove, cluster_duplicates_dfs = semantic_deduplication(
            df = df,
            required_columns = ["problem_answer"],
            num_kmeans_clusters = len(df) // 100,
            embedding_batch_size = 2500,
            use_gpu=True
        )

        # add the semdedup outcome to the dataset
        partial_is_semdedup_duplicate = partial(is_semdedup_duplicate, indices_to_remove=indices_to_remove, epsilon=epsilon)
        dataset = dataset.map(partial_is_semdedup_duplicate, with_indices=True, num_proc=mp.cpu_count())

        print(f"Epsilon {epsilon} Filtered Dataset: {dataset}")
        print(f"Epsilon {epsilon} Number of duplicates: {len(indices_to_remove)}")

    # remove the problem_answer column
    dataset = dataset.remove_columns("problem_answer")

    # push the merged dataset to the hub
    dataset.push_to_hub(DATASET_PATH, private=True)

if __name__ == "__main__":
    main()