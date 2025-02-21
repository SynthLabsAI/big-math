import asyncio
import json
import math
import pprint
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

import dspy

load_dotenv()

from modules import MathProblemReformulator

# Global configs
# MODEL = "openai/gpt-4-turbo-preview"  # default model
MODEL = "openai/meta-llama/Llama-3.1-8B-Instruct"
CONCURRENT_TASKS_PER_API = 25

# Server configurations
API_CONFIGS = {
    """Example"""
    "<Endpoint IP Address>": {
        "api_base": "http://<Endpoint IP Address>:<port>/v1",
        "api_key": "PROVIDER_API_KEY",
    },
    "123.456.789.012": {
        "api_base": "http://123.456.789.012:8000/v1",
        "api_key": "PROVIDER_API_KEY",
    },
}


def custom_load_dataset():
    """
    This function is a placeholder for loading a custom dataset.

    You need to define this function based on the specific requirements of your dataset.
    It should include the logic to load and return the dataset from its source.

    Returns:
        list: An empty list as a placeholder. Replace this with the actual dataset.
    """
    """Define your own dataset loader here, depending on where the data comes from"""
    return []

def setup_dspy_with_endpoint(api_base: str, api_key: str, temperature: float = 0.0001) -> None:
    """Configure DSPy with specified endpoint and settings"""
    lm = dspy.LM(
        MODEL,
        api_key=api_key,
        api_base=api_base,
        temperature=round(temperature, 6),
        # model_type='chat'
    )
    # Configure async worker pool size
    dspy.settings.configure(lm=lm, async_max_workers=CONCURRENT_TASKS_PER_API)
    logger.debug(f"DSPy configured with model: {MODEL} at {api_base}")

def save_batch_results(batch_data: List[Dict[str, Any]], output_dir: str = "outputs") -> None:
    """Append batch results to a single JSONL file"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a single JSONL file for all results
    output_file = output_dir / f"reformulation_results.jsonl"

    # Append results to file
    with open(output_file, "a") as f:
        for item in batch_data:
            f.write(json.dumps(item) + "\n")

    logger.debug(f"Batch results appended to {output_file}")


def load_processed_problems(output_dir: str = "outputs") -> set:
    """Load set of already processed problem IDs that passed judge verification"""
    output_files = list(Path(output_dir).glob(f"reformulation_results_*.jsonl"))
    processed = set()

    if output_files:
        for output_file in output_files:
            with open(output_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Only consider it processed if it was successful AND got a pass verdict
                        if (data.get('success') and
                            'data' in data and
                            data.get('data', {}).get('judge_verdict', '').lower() == 'pass'):
                            processed.add(data['data']['uuid'])
                    except:
                        continue

    logger.info(f"Found {len(processed)} previously processed problems with PASS verdict")
    return processed


def log_processing_details(problem: str, result) -> None:
    """Log the processing details for a single item"""
    # Print and log results with clear section headers
    logger.info("=== Original Problem ===")
    print("\n=== Original Problem ===")
    logger.debug(problem)
    print(problem)

    # Display reformulation process details
    key_display_map = {
        "core_mathematical_concept": "<fill this in>",
        "key_information_extraction": "<fill this in>",
        "problem_structure_analysis": "<fill this in>",
        "multiple_choice_removal_strategy": "<fill this in>",
        "rephrasing_approach": "<fill this in>",
        "problem_integrity_preservation": "<fill this in>",
        "answer_format_specification": "<fill this in>"
    }

    logger.info("=== Reformulation Process Details ===")
    print("\n=== Reformulation Process Details ===")
    for key, value in result.reformulation_process.items():
        display_key = key_display_map.get(key, key.replace('_', ' ').title())
        logger.debug(f"{display_key}: {value}")
        print(f"{display_key}:")
        print(f"  {value}")

    logger.info("=== Reformulated Problem ===")
    print("\n=== Reformulated Problem ===")
    logger.debug(result.reformulated_problem)
    print(result.reformulated_problem)

    # Log judge results
    logger.info("=== Judge Evaluation ===")
    print("\n=== Judge Evaluation ===")
    print(f"Verdict: {result.judge_verdict}")

    if result.judge_issues:
        print("\nIssues Found:")
        for issue in result.judge_issues:
            print(f"- {issue}")

    if result.judge_suggestions:
        print("\nSuggestions:")
        for suggestion in result.judge_suggestions:
            print(f"- {suggestion}")

    if result.judge_corrected_reformulated_problem:
        logger.info("=== Judge's Corrected Version ===")
        print("\n=== Judge's Corrected Version ===")
        print(result.judge_corrected_reformulated_problem)

    # Log validation results
    logger.info("=== Basic Validation Results ===")
    print("\n=== Basic Validation Results ===")
    print("Checking the following criteria:")
    validation_descriptions = {
        "no_mc_options": "Multiple choice options removed",
        "process_complete": "All reformulation steps completed"
    }

    for check, passed in result.validation_details.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        description = validation_descriptions.get(check, check)
        logger.debug(f"{status} | {description}")
        print(f"{status} | {description}")

    overall = "✓ PASSED" if result.validation_result else "✗ FAILED"
    logger.info(f"Overall Basic Validation: {overall}")
    print(f"\nOverall Basic Validation: {overall}")
    print("\n" + "="*50 + "\n")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def process_single_item(item: Dict[str, Any], reformulator, api_base: str = None) -> Dict[str, Any]:
    """Process a single item and return its result"""
    try:
        problem_text = item['problem']
        if not problem_text:
            raise ValueError("Empty problem text")
        try:
            correct_answer = item['final_answer']
        except:
            logger.warning("No final answer found")
            correct_answer = None
        if not correct_answer:
            try:
                correct_answer = item['answer']
            except:
                logger.warning("No answer found")
                correct_answer = None
        if not correct_answer:
            try:
                correct_answer = item['source_solution']
            except:
                logger.warning("No final answer found")
                correct_answer = None
        if not correct_answer:
            try:
                correct_answer = item['solution']
            except:
                logger.warning("No solution found")
                pprint.pprint(item)
                # input("NO SOLUTION/ANSWER FOUND!!! YIKKKKEESSSS...... Press Enter to continue...")
                correct_answer = None
        print(f"Correct answer: {correct_answer}")
        # Add uuid if not present for single problems
        if 'uuid' not in item:
            item['uuid'] = str(datetime.now().timestamp())

        temperature = round(random.uniform(0.00001, 0.001), 6)

        # Call reformulator asynchronously with randomized temperature
        with dspy.context(
            lm=dspy.LM(
                MODEL,
                temperature=temperature,
                api_base=api_base,
                api_key="PROVIDER_API_KEY",
                # model_type='chat'
            )):
            result = await reformulator(problem_text, correct_answer)

        processed_data = {
            'reformulation_process': result.reformulation_process,
            'reformulated_problem': result.reformulated_problem,
            'reformulation_solution': result.reformulation_solution,
            'validation_details': result.validation_details,
            'validation_result': result.validation_result,
            'reasoning': result.reasoning if result.reasoning else None,
            'judge_verdict': result.judge_verdict,
            'judge_issues': result.judge_issues,
            'judge_suggestions': result.judge_suggestions,
            'judge_reasoning': result.judge_reasoning,
            'judge_corrected_reformulated_problem': result.judge_corrected_reformulated_problem,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'temperature': temperature,
            'api_base': api_base,
            **item,
        }

        logger.info("=== Processing Results ===")
        log_processing_details(problem_text, result)

        return {"success": True, "data": processed_data}
    except Exception as e:
        logger.error(f"Failed to process problem: {e}")
        return {"success": False, "error": str(e), "item": item}

async def process_batch_distributed(items, reformulator, api_resources: Dict) -> List[Dict[str, Any]]:
    """Process a batch of items distributed across multiple API endpoints"""
    new_items = [item for item in items if item['uuid']]

    if not new_items:
        return []

    api_endpoints = list(api_resources.keys())
    num_endpoints = len(api_endpoints)

    tasks = []
    for idx, item in enumerate(new_items):
        api_base = api_endpoints[idx % num_endpoints]
        resources = api_resources[api_base]
        pprint.pprint(item)
        setup_dspy_with_endpoint(
            api_base=resources['api_base'],
            api_key=resources['api_key'],
            temperature=round(random.uniform(0.00001, 0.001), 6)
        )

        task = process_single_item(item, reformulator, api_base=resources['api_base'],)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    successful_results = [r for r in results if r["success"]]
    if successful_results:
        save_batch_results(successful_results)

    return results

async def main(problem: Optional[str] = None) -> None:
    """Main function to run math problem reformulation"""
    # Initialize API resources
    api_resources = {
        api_base: {
            'api_base': config['api_base'],
            'api_key': config['api_key']
        } for api_base, config in API_CONFIGS.items()
    }

    # Create reformulator instance and make it async
    reformulator = dspy.asyncify(MathProblemReformulator())
    logger.debug("MathProblemReformulator instance created")

    if problem is None:
        # Load and process dataset in parallel
        dataset = custom_load_dataset()
        if not dataset:
            logger.error("No dataset loaded. Exiting.")
            return

        total_problems = len(dataset)
        logger.info(f"Processing {total_problems} problems from dataset")

        print("\nSample row from dataset:")
        first_item = dataset[0]
        print(json.dumps(first_item, indent=2))
        # input("\nPress Enter to continue...")

        batch_size = len(api_resources) * CONCURRENT_TASKS_PER_API
        with tqdm(total=total_problems, desc="Processing problems") as pbar:
            for i in range(0, total_problems, batch_size):
                batch = dataset[i:i + batch_size]

                logger.info(f"Processing batch {i//batch_size + 1} of {math.ceil(total_problems/batch_size)}")
                results = await process_batch_distributed(batch, reformulator, api_resources)
                pbar.update(len(results))
    else:
        # Process single problem
        api_base = next(iter(API_CONFIGS.keys()))
        config = API_CONFIGS[api_base]
        setup_dspy_with_endpoint(config['api_base'], config['api_key'])

        try:
            # Create item dictionary matching dataset format
            item = {
                "problem": problem,
                "source": "manual_input"
            }
            result = await process_single_item(item, reformulator)
            if result["success"]:
                save_batch_results([result])
                logger.info("Successfully saved single problem result")
        except Exception as e:
            logger.error(f"Failed to process single problem: {e}")

if __name__ == "__main__":
    asyncio.run(main())