from datasets import load_dataset
import multiprocessing as mp
import re
import argparse

def detect_latex_hyperlinks(row):
    # Check for \url{} links
    url_pattern = r'\\url'
    # Check for \href{}{} links
    href_pattern = r'\\href'
    # Check for http:// or https:// links
    http_pattern = r'http://'
    https_pattern = r'https://'
    
    problem = row['problem']

    # Search for patterns
    if re.search(url_pattern, problem) or re.search(href_pattern, problem) \
        or re.search(http_pattern, problem) or re.search(https_pattern, problem):
        row['has_hyperlink_problem'] = True
    else:
        row['has_hyperlink_problem'] = False

    solution = row['solution']

    # Search for patterns
    if re.search(url_pattern, solution) or re.search(href_pattern, solution) \
        or re.search(http_pattern, solution) or re.search(https_pattern, solution):
        row['has_hyperlink_solution'] = True
    else:
        row['has_hyperlink_solution'] = False

    return row

def main(dataset_path):
    # load the dataset
    dataset = load_dataset(dataset_path, split="train")

    # run hyperlink detection over the full dataset
    dataset = dataset.map(detect_latex_hyperlinks, num_proc=mp.cpu_count())

    # push the updated dataset
    dataset.push_to_hub(dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect LaTeX hyperlinks in a dataset.")
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    main(args.dataset_path)