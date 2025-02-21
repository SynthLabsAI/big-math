import argparse
import asyncio
from datasets import Dataset, load_dataset
from enum import Enum
import json
import os
from pydantic import BaseModel, Field, field_validator
import tenacity
import traceback
from tqdm import tqdm

from utils.sglang_util import SGLangServerManager

class ModelType(str, Enum):
    """supported llm model types"""
    Llama3_1_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    Llama3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    Llama3_1_405B = "meta-llama/Llama-3.1-405B-Instruct-FP8"
    CLAUDE_3_5 = "claude-3-5-sonnet-latest"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"

class LLMHyperparams(BaseModel):
    system_prompt: str = """You are a math expert. Given the following math problem, provide your solution in Latex format. Always format your final answer in perfect LaTeX \\boxed{{final_answer}} format."""
    prompt: str = "{problem}"
    temperature: float = Field(
        default=0.8,
        ge=0.0, 
        le=2.0,
        description='Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.'
    )
    top_k: int = Field(
        default=-1,
        description='Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.'
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description='Float that controls the cumulative probability of the top tokens to consider.'
    )
    max_tokens: int = 2048
    model_name: ModelType = Field(
        description='The model to use for generation.',
    )
    n: int = Field(
        description='Number of samples to generate.'
    )

    @field_validator('temperature')
    def validate_temperature(cls, v):
        if v < 0:
            raise ValueError("temp can't be negative")
        return v

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    @classmethod
    def load_json(cls, json_str: str) -> 'LLMHyperparams':
        data = json.loads(json_str)
        return cls(**data)

def construct_few_shot_prompt(problem):
    # Updated prompt as per the user's instructions
    system_prompt = r"""From this moment forward, adopt the persona of a tenured mathematics professor. You receive math problems and you solve them, step by step, following the formatting instructions below.

# Mathematical Solution Formatting Guidelines

## Mathematical Notation Rules

**Use $$ $$ for:**
- Standalone equations
- Complex mathematical expressions
- Multi-line equations
- Final answers

**Use single $ $ for:**
- Simple inline expressions
- Variables
- Numbers with mathematical meaning
- Parts of expressions being discussed

**For equations with multiple lines:**
`latex
$$\\begin{aligned}
equation1 &= expression1 \\\\
equation2 &= expression2
\\end{aligned}$$
`

**Place the final answer inside \boxed{answer_here}**
After solving and verifying your full solution, write the final answer in \\boxed{answer} notation.

## Thought Structure
Each thought must:
1. Begin with active voice ("I" statements or direct observations)
2. Express exactly ONE logical step
3. Integrate mathematical notation naturally within sentences
4. Use appropriate notation based on context
5. No more than 1 sentence + interleaved math notation long

### Example:
"I can see that when $x = 2$, the expression $$y = x^2 + 3x + 1$$ becomes $$y = 4 + 6 + 1 = 11$$."

## Common Mathematical Patterns
- Use $$ $$ for equations being solved
- Use $ $ for discussing components: "where $m$ is the slope"
- Keep expressions inline when discussing specific parts
- Use block notation ($$) for key steps and results

## Visual Formatting
- Two blank lines between thoughts (\n\n)
- No bullet points or numbered lists
- No section headers within the solution
- Mathematical expressions should flow naturally within sentences
- No explicit step labeling
- Human-style reasoning

---

## Examples

### Problem 1:

A standard die is rolled six times. What is the probability that the product of all six rolls is odd? Express your answer as a common fraction.

### Solution 1:

I need to find the probability by dividing favorable outcomes by total outcomes.

I know that the total number of possible outcomes when rolling a die six times is $$6^6$$.

For the product to be odd, each individual roll must be odd.

Looking at a standard die, the odd numbers are 1, 3, and 5.

Therefore, for each roll, I have only 3 choices to maintain an odd product.

The total number of favorable outcomes is $$3^6$$ since I have 3 choices for each of the 6 rolls.

The probability is thus $$\frac{3^6}{6^6}$$.

This simplifies to $$\left( \frac{1}{2} \right)^6 = \\boxed{\frac{1}{64}}$$.

### Problem 2:

Solve over the integers: $$2^a+4^b+8^c=328$$

### Solution 2:

I notice that all terms are powers of 2, so I can rewrite everything with base 2.

I know that $$4 = 2^2$$ and $$8 = 2^3$$, so the equation becomes $$2^a + 2^{2b} + 2^{3c} = 328$$.

To solve this, I can decompose 328 into powers of 2.

The largest power of 2 less than 328 is 256 ($$2^8$$).

Subtracting 256 from 328 leaves 72.

The largest power of 2 less than 72 is 64 ($$2^6$$).

Subtracting 64 leaves 8 ($$2^3$$).

Therefore, $$328 = 2^8 + 2^6 + 2^3$$.

Comparing terms: $$2^a = 2^8$$, $$2^{2b} = 2^6$$, and $$2^{3c} = 2^3$$.

Solving these equations: $$a = 8$$, $$b = 3$$, and $$c = 1$$.

My final answer is \\boxed{(a, b, c) = (8, 3, 1)}

### Problem 3:

Find the equation of the circle which passes through $$(2, 3)$$ and $$(4, 5)$$ and whose center lies on the straight line $$y - 4x + 3 = 0$$.

### Solution 3:

I know that the center lies on the line, so I can write its coordinates as $$(x, 4x - 3)$$.

The radius can be found using the distance from the center to $$(2, 3)$$.

Using the distance formula, I get $$r^2 = (x - 2)^2 + (4x - 3 - 3)^2 = (x - 2)^2 + (4x - 6)^2$$.

Expanding and simplifying: $$r^2 = 17x^2 - 52x + 40$$.

Since the circle also passes through $$(4, 5)$$, I can write $$r^2 = (x - 4)^2 + (4x - 3 - 5)^2$$.

This simplifies to $$r^2 = 17x^2 - 72x + 80$$.

Since both expressions equal $$r^2$$, I can write $$17x^2 - 52x + 40 = 17x^2 - 72x + 80$$.

I can now simplify: $$20x = 40$$, so $$x = 2$$.

The center is therefore $$(2, 4(2) - 3) = (2, 5)$$.

The radius squared is $$r^2 = 17(2)^2 - 52(2) + 40 = 16$$, so $$r = 4$$.

The equation of the circle is $$(x - 2)^2 + (y - 5)^2 = 16$$.

The final answer is the full equation of the circle, so the correct final answer is \\boxed{(x - 2)^2 + (y - 5)^2 = 16}

"""

    # Append the real example as a user/assistant message
    user_problem = r"Given real numbers \( a, b, c \) and a positive number \( \lambda \) such that the polynomial \( f(x) = x^3 + a x^2 + b x + c \) has three real roots \( x_1, x_2, x_3 \), and the conditions \( x_2 - x_1 = \lambda \) and \( x_3 > \frac{1}{2}(x_1 + x_2) \) are satisfied, find the maximum value of \( \frac{2 a^3 + 27 c - 9 a b}{\lambda^3} \)."

    assistant_solution = r"""
I need to find the maximum value of the given expression, which involves the coefficients of the polynomial and the difference between its roots.

Since \( x_1, x_2, x_3 \) are the roots of the polynomial \( f(x) = x^3 + a x^2 + b x + c \), by Vieta's formulas, I know that $$x_1 + x_2 + x_3 = -a$$.

Also from Vieta's formulas, $$x_1 x_2 + x_2 x_3 + x_3 x_1 = b$$ and $$x_1 x_2 x_3 = -c$$.

Given that \( x_2 - x_1 = \lambda \), I can express \( x_2 \) as \( x_2 = x_1 + \lambda \).

The condition \( x_3 > \frac{1}{2}(x_1 + x_2) \) implies that \( x_3 > x_1 + \frac{\lambda}{2} \).

However, to simplify the problem, I will consider the equality case, which should give me the maximum value.

Thus, I set \( x_3 = x_1 + \frac{\lambda}{2} \).

Now, I substitute \( x_2 \) and \( x_3 \) in terms of \( x_1 \) and \( \lambda \) into the expression for \( a \): $$-a = x_1 + (x_1 + \lambda) + \left( x_1 + \frac{\lambda}{2} \right ).$$

This simplifies to $$-a = 3x_1 + \frac{3\lambda}{2}.$$

I can also express \( b \) and \( c \) in terms of \( x_1 \) and \( \lambda \):
$$b = x_1(x_1 + \lambda) + (x_1 + \lambda)\left( x_1 + \frac{\lambda}{2} \right ) + \left( x_1 + \frac{\lambda}{2} \right ) x_1,$$
and
$$-c = x_1(x_1 + \lambda)\left( x_1 + \frac{\lambda}{2} \right ).$$

Substituting these expressions into the given expression, I get $$\frac{2 a^3 + 27 c - 9 a b}{\lambda^3}$$ in terms of \( x_1 \) and \( \lambda \).

However, since I want the maximum value of this expression, which is independent of \( x_1 \), I can choose a specific value of \( x_1 \) to simplify the calculation.

Let's set \( x_1 = 0 \) to get the maximum value.

Substituting \( x_1 = 0 \) into the expressions for \( a \), \( b \), and \( c \), I get
$$-a = \frac{3\lambda}{2},$$
$$b = \frac{\lambda^2}{2},$$
and
$$-c = 0.$$

Now, I substitute these values into the given expression:
$$\frac{2 a^3 + 27 c - 9 a b}{\lambda^3} = \frac{2\left( -\frac{3\lambda}{2} \right )^3 + 27(0) - 9\left( -\frac{3\lambda}{2} \right )\left( \frac{\lambda^2}{2} \right )}{\lambda^3}.$$

Simplifying the numerator:
$$2\left( -\frac{27\lambda^3}{8} \right ) + \frac{27\lambda^3}{4} = -\frac{27\lambda^3}{4} + \frac{27\lambda^3}{4} = 0.$$

Therefore, the maximum value of the expression is $$\frac{0}{\lambda^3} = 0.$$

\\boxed{0}
"""

    # Prepare the messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_problem},
        {"role": "assistant", "content": assistant_solution},
        {"role": "user", "content": f"Problem: {problem}"},
    ]
    return messages

def format_items(problem, system_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]

@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
async def get_chat_responses(server_handler, batch_items, n, top_p, temperature, max_tokens):
    return await server_handler.get_chat_responses(
        batch_items,
        n=n, 
        top_p=top_p, 
        temperature=temperature,
        max_tokens=max_tokens
    )

async def main():
    # Configuration
    model_name = args.model_name
    dataset_name = args.dataset_name
    save_name = args.save_name
    save_folder = args.save_folder
    problem_column_name = args.problem_column_name
    response_column_name=args.response_column_name
    if args.dataset_name_outputs:
        hf_save_dataset_name = args.dataset_name_outputs
    else:
        hf_save_dataset_name = dataset_name
    os.makedirs(save_folder, exist_ok=True)

    batch_size = args.batch_size
    save_interval = args.save_interval

    # Sampling hyperparameters
    if args.greedy:
        # greedy decoding
        llm_params = LLMHyperparams(
            temperature=0.0, 
            model_name=ModelType(model_name), 
            n=1,
            max_tokens=args.max_tokens
        )
    else:
        llm_params = LLMHyperparams(
            top_k=args.top_k, 
            top_p=args.top_p, 
            temperature=args.temperature, 
            model_name=ModelType(model_name), 
            n=args.n,
            max_tokens=args.max_tokens
        )
    
    print("Hyperparameters:")
    print(llm_params.to_json())

    # Load and preprocess dataset
    ds = load_dataset(dataset_name, split=args.dataset_split)
    df = ds.to_pandas()

    # if only using partial dataset, slice it
    if args.end == -1:
        args.end = len(df)
    df = df.iloc[args.start:args.end]
    df = df.reset_index(drop=True)

    print(f"Total dataset: {len(df)}")

    # Prepare items for processing
    if args.generate_training_data:
        items = [construct_few_shot_prompt(row[problem_column_name]) for _, row in df.iterrows()]
    else:
        items = [format_items(row[problem_column_name], llm_params.system_prompt) for _, row in df.iterrows()]

    df[response_column_name] = None
    # Process items in batches
    total_items = len(items)
    count = 0
    with SGLangServerManager(model_name, tp=args.tp) as server_handler:
        for idx in tqdm(range(0, total_items, batch_size)):
            batch_items = items[idx:idx+batch_size]
            batch_outputs = await get_chat_responses(
                server_handler=server_handler,
                batch_items=batch_items,
                n=llm_params.n, 
                top_p=llm_params.top_p, 
                temperature=llm_params.temperature,
                max_tokens=llm_params.max_tokens
            )
            batch_responses = []
            for resp in batch_outputs:
                try:
                    batch_responses.append(resp[-1]["responses"])
                except Exception as e:
                    print(f"Response: {resp}")
                    traceback_str = traceback.format_exc()
                    print(f"Error processing response: {traceback_str}")
                    batch_responses.append([""])

            count += 1
            # Assign responses to dataframe
            for i, response_list in enumerate(batch_responses):
                df.at[idx+i, response_column_name] = response_list
            if count % save_interval == 0:
                try:
                    df.iloc[:idx+batch_size].to_parquet(
                        os.path.join(save_folder, f"{save_name}_{count}_batch.parquet")
                    )
                    ds = Dataset.from_pandas(df)
                    ds.push_to_hub(hf_save_dataset_name, private=True)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
    # Save final results
    try:
        df.to_parquet(os.path.join(save_folder, f"{save_name}.parquet"))
        ds = Dataset.from_pandas(df)
        ds.push_to_hub(hf_save_dataset_name, private=True)
        print(f"Saved to {os.path.join(save_folder, f'{save_name}.parquet')}")
    except Exception as e:
        print(f"Error saving final results: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Model and dataset configuration
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_name_outputs", type=str,
                        help="To save the outputs to a different HF dataset, specify here.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--problem_column_name", type=str, default="problem")
    parser.add_argument("--response_column_name", type=str, required=True)
    parser.add_argument("--generate_training_data", action="store_true",
                        help="Use a few-shot prompt to encourage the model to follow a specific format.")
    
    # Save configuration
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="Save every n batches.")

    # SGLang server configuration
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=250,
                        help="Total batch size will be args.batch_size * args.n.")

    # LLM Hyperparameters
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding. Ignores temperature, top_p, and top_k.")

    # dataset slicing
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    asyncio.run(main())

# Example Usage:
# python3 sample_from_model.py --model_name meta-llama/Llama-3.1-405B-Instruct-FP8 --dataset_name RLAIF/Big-Math-Competition-Problems --save_folder math_competition_problems --save_name llama405b_greedy --response_column_name llama_405b_greedy_response --greedy --tp 8

# python3 sample_from_model.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --dataset_name RLAIF/Big-Math-Competition-Problems --save_folder math_competition_problems --save_name llama8b_greedy --response_column_name llama_8b_greedy_response --greedy