from datasets import load_dataset
import fasttext
from functools import partial
from huggingface_hub import hf_hub_download
import multiprocessing as mp
import re
import argparse

# load the fasttext language identification model
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
fasttext_lang_id_model = fasttext.load_model(model_path)

def remove_special_chars(text):
    # Remove numbers, special characters, and math symbols
    pattern = r'[0-9*,\.+\-=\(\)\/\^\[\]{}|<>~`!@#$%&?_]'
    text = re.sub(pattern, '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_latex_commands(text):
    # remove \frac{}{} commands, and any arbitrary text inside the curly braces
    text = re.sub(r'\\frac\{[^{}]*\}\{[^{}]*\}', '', text)

    # Remove commands with arguments
    text = re.sub(r'\\[a-zA-Z]+\{[^{}]*\}', '', text)
    
    # Remove standalone commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # Remove math environments
    text = re.sub(r'\$\$(.*?)\$\$', '', text)
    text = re.sub(r'\$(.*?)\$', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_text(text: str):
    preds = fasttext_lang_id_model.predict(text, k=-1)
    return [(label.replace("__label__", ""), float(score)) for label, score in zip(*preds)]

def detect_lang(row, column_name='problem'):
    if not row[column_name]:
        row[f'fasttext_lang_{column_name}'] = []
        row[f'{column_name}_language'] = "No language detected"
        return row

    cleaned_text = remove_special_chars(remove_latex_commands(row[column_name]).replace("\n", ""))
    lang_probs = predict_text(cleaned_text)

    # reformat the lang_probs into list of strings
    row[f'fasttext_lang_{column_name}'] = [f"{lang}: {prob}" for lang, prob in lang_probs[:5]]

    # if the text is too short, just allow it
    if len(cleaned_text) < 10:
        row[f'{column_name}_language'] = "en"
        return row
    
    # set default language
    row[f'{column_name}_language'] = "en"

    for lang, prob in lang_probs:
        if lang == "eng_Latn":
            eng_prob = prob
            
    # label anything with >2% probability of english as english
    if eng_prob < 0.02:
        highest_prob_lang = lang_probs[0][0]
        # if highest likelihood language uses latin character, set to "en"
        if highest_prob_lang.endswith("_Latn"):
            row[f'{column_name}_language'] = "en"
        else:
            row[f'{column_name}_language'] = highest_prob_lang

    return row

def main(dataset_path):
    # define the columns to detect language on
    detect_language_on = ["problem", "final_answer"]
    new_columns = [f'{col}_language' for col in detect_language_on]
    new_columns += [f'fasttext_lang_{col}' for col in detect_language_on]

    # load the dataset
    dataset = load_dataset(dataset_path, split="train")

    for col in detect_language_on:
        # create function on the "problem" column
        detect_language = partial(detect_lang, column_name=col)

        # run language detection over the full dataset
        dataset = dataset.map(detect_language, num_proc=mp.cpu_count())

    # push the updated dataset
    dataset.push_to_hub(dataset_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect language in a dataset.")
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    args = parser.parse_args()
    main(args.dataset_path)