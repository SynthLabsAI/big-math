import argparse
import asyncio
from datasets import Dataset, load_dataset
from enum import Enum
import json
import os
from pydantic import BaseModel, Field, field_validator
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

no_multiple_parts = [
    """\
Ancient Greek mathematicians from the Pythagorean school studied various polygonal numbers, such as triangular numbers 1, 3, 6, 10, ..., with the $n$-th triangular number being $\\frac{n(n+1)}{2} = \\frac{1}{2}n^{2} + \\frac{1}{2}n$. Let the $n$-th $k$-sided polygon number be denoted as $N(n, k)$ ($k \\geq 3$). Below are the expressions for the $n$-th number of some $k$-sided polygon numbers:  \nTriangular numbers $N(n, 3) = \\frac{1}{2}n^{2} + \\frac{1}{2}n$  \nSquare numbers $N(n, 4) = n^{2}$  \nPentagonal numbers $N(n, 5) = \\frac{3}{2}n^{2} - \\frac{1}{2}n$  \nHexagonal numbers $N(n, 6) = 2n^{2} - n$  \n...  \nFrom this, we can deduce the expression for $N(n, k)$ and calculate $N(8, 12) = \\_\\_\\_\\_\\_\\_$.'
""",
    """\
Let x be the dividend, y be the divisor, z be the quotient, and r be the remainder. If y = 3(y1 + y2) + 4, z = 2z1^2 - z2, r = 3r1 + 2, and x = 2x1y1 - x2 + 10, find the values of x, y, z, and r, given that y1 = 2, y2 = 3, z1 = 3, z2 = 5, r1 = 1, x1 = 4, and x2 = 6.
""",
    """\
If  $x_{1}, x_{2},\ldots ,x_{n}$  are positive real numbers with  $x_{1}^2+x_2^{2}+\ldots +x_{n}^{2}=1$ , ﬁnd the minimum value of  $\sum_{i=1}^{n}\frac{x_{i}^{5}}{x_{1}+x_{2}+\ldots +x_{n}-x_{i}}$ .
""",
    """
Given that the value of the function \( f(x) = \frac{1}{(x-2)^{2}} - 2x + \cos 2\theta - 3 \sin \theta + 2 \) is always positive for \( x \in (-\infty, 2) \), determine the range of the parameter \( \theta \) within the interval \( (0, \pi) \).
""",
    """
A positive integer \( n \) is written on the blackboard. Two players, A and B, take turns performing operations, starting with A. On A's turn, if the number on the board is \( k \), A replaces it with one of \( \frac{k}{2} \), \( \frac{k}{4} \), or \( 3k \) (with the first two choices being allowed only if \( \frac{k}{2} \) or \( \frac{k}{4} \) are integers). On B's turn, if the number on the board is \( k \), B replaces it with either \( k+1 \) or \( k-1 \). Player A wins the game if the number 3 is written on the blackboard at any point. For which positive integers \( n \) does player A have a winning strategy?
""",
    """\
A student was studying the properties of the function $f(x) = x^2e^x$ and came to the following conclusions:
- ① The interval where $f(x)$ is monotonically decreasing is $(-2, 0)$;
- ② $f(x)$ has neither a minimum nor a maximum value;
- ③ The graph of $f(x)$ and its tangent line at $(0,0)$ intersect at two points;
- ④ The graph of $f(x)$ and the line $x - y + 2012 = 0$ intersect at two points.

Among these conclusions, the correct ones are __________.
""",
    """\
Prove that the set \(A=\left\{2,2^{2}, \cdots\right., \left.2^{n}, \cdots\right\}\) satisfies the following:

(1) For every \(a \in A\) and \(b \in \mathbf{N}^{*}\), if \(b < 2a - 1\), then \(b(b+1)\) is not a multiple of \(2a\).

(2) For every \(a \in \bar{A}\) (where \(\bar{A}\) denotes the complement of \(A\) in \(\mathbf{N}^{*}\)) and \(a \neq 1\), there exists \(b \in \mathbf{N}^{*}\) such that \(b < 2a - 1\) and \(b(b+1)\) is a multiple of \(2a\).
""",
    """\
You have a 200 liters mixture of four chemicals: W, X, Y, and Z which are in the ratio of 3:4:6:7. You add 40 liters of chemical W, 35 liters of chemical X, 50 liters of chemical Y and 25 liters of chemical Z to this mixture. What is the new percentage of each chemical W, X, Y and Z in the resulting mixture?
""",
    """\
Which of the following statements are correct?

A: If a sample of size $5$ is drawn from a population of $50$ individuals using simple random sampling, the probability of individual $m$ being selected is $0.1$.

B: Given a set of data $1$, $2$, $m$, $6$, $7$ with an average of $4$, then the variance of this data set is $\frac{{26}}{5}$.

C: The $70$th percentile of the data $13$, $27$, $24$, $12$, $14$, $30$, $15$, $17$, $19$, $23$ is $23$.

D: If the standard deviation of a sample data $x_{1}$, $x_{2}$, $\cdots$, $x_{10}$ is $8$, then the standard deviation of the data $2x_{1}-1$, $2x_{2}-1$, $\cdots$, $2x_{10}-1$ is $32$.
""",
    """\
A transparent, sealed cubic container is exactly half filled with water. When this cube is rotated arbitrarily, the shape of the water surface inside the container can be: (1) triangle; (2) rectangle; (3) square; (4) regular hexagon. Among these, the correct conclusions are __________.
""",
    """\
Given a 100-sided polygon \( P \) in the Cartesian coordinate plane, satisfying:
(i) The coordinates of the vertices of \( P \) are all integers;
(ii) The sides of \( P \) are parallel to the coordinate axes;
(iii) The side lengths of \( P \) are all odd numbers.

Prove that the area of \( P \) is an odd number.
""",
    """\
Given a positive integer $n \ge 2$. Find all $n$-tuples of positive integers $(a_1,a_2,\ldots,a_n)$, such that $1<a_1 \le a_2 \le a_3 \le \cdots \le a_n$, $a_1$ is odd, and
(1) $M=\frac{1}{2^n}(a_1-1)a_2 a_3 \cdots a_n$ is a positive integer;
(2) One can pick $n$-tuples of integers $(k_{i,1},k_{i,2},\ldots,k_{i,n})$ for $i=1,2,\ldots,M$ such that for any $1 \le i_1 <i_2 \le M$, there exists $j \in \{1,2,\ldots,n\}$ such that $k_{i_1,j}-k_{i_2,j} \not\equiv 0, \pm 1 \pmod{a_j}$.
""",
    """\
The numbers $1,2,\ldots,64$ are written in the squares of an $8\times 8$ chessboard, one number to each square. Then $2\times 2$ tiles are placed on the chessboard (without overlapping) so that each tile covers exactly four squares whose numbers sum to less than $100$. Find, with proof, the maximum number of tiles that can be placed on the chessboard, and give an example of a distribution of the numbers $1,2,\ldots,64$ into the squares of the chessboard that admits this maximum number of tiles.
""",
    """\
For any integer $k$, write $f_{k}(x)=\left(1+x^{2}\right)^{-1-k}$. When $k \geq 1$, find constants $c_{1}, c_{2}$ such that the function $y=\left(S f_{k}\right)(x)$ solves a second order differential equation $x y^{\prime \prime}+c_{1} y^{\prime}+c_{2} x y=0$.
""",
]
multiple_parts = [
    """\
Given the set $M=\{0,1\}$, $A=\{(x,y)|x\in M, y\in M\}$, $B=\{(x,y)|y=-x+1\}$. 1. Please list the elements of set $A$. 2. Find $A\cap B$ and list all subsets of $A\cap B$.
""",
    """\
In the Cartesian coordinate system $xOy$, the parametric equation of curve $C_1$ is $\begin{cases} x=\cos \theta \\ y=1+\sin \theta \end{cases}$ (where $\theta$ is the parameter), and the equation of curve $C_2$ is $\frac{x^2}{1}+\frac{y^2}{2}=1$. With $O$ as the pole and the non-negative half-axis of $x$ as the polar axis, a polar coordinate system is established with the same unit of length as the Cartesian coordinate system $xOy$. (1) Find the polar equations of curves $C_1$ and $C_2$; (2) The ray $\theta =\frac{\pi }{3}(\rho > 0)$ intersects curve $C_1$ at point $A$ (other than the pole) and intersects curve $C_2$ at point $B$. Find $|AB|$.
""",
    """\
Given the function $f(x)=|x+2|-|2x-a|$, $(a\in\mathbb{R})$. (I) When $a=3$, solve the inequality $f(x) > 0$; (II) When $x \in [0, +\infty)$, $f(x) < 3$ always holds, find the range of $a$.
""",
    """\
Given an ellipse $C_1$: $\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$ ($a > b > 0$) with a major axis length of 4 and an eccentricity of $\frac{1}{2}$, where $F_1$ and $F_2$ are its left and right foci, respectively. A moving circle passes through point $F_2$ and is tangent to the line $x = -1$. (Ⅰ) (i) Find the equation of the ellipse $C_1$; (ii) Find the equation of the trajectory of the center $C$ of the moving circle; (Ⅱ) On the curve $C$, there are two points $M$ and $N$, and on the ellipse $C_1$, there are two points $P$ and $Q$, satisfying that $MF_2$ and $\overrightarrow{NF_2}$ are collinear, $\overrightarrow{PF_2}$ and $\overrightarrow{QF_2}$ are collinear, and $\overrightarrow{PF_2} \cdot \overrightarrow{MF_2} = 0$, find the minimum value of the area of quadrilateral $PMQN$.
""",
    """\
In the rectangular coordinate system $xOy$, a polar coordinate system is established with the coordinate origin as the pole and the positive semi-axis of the $x$-axis as the polar axis. The polar coordinate equation of circle $C$ is $\rho^2 - 2m\rho\cos\theta + 4\rho\sin\theta = 1 - 2m$. (1) Find the rectangular coordinate equation of $C$ and its radius. (2) When the radius of $C$ is the smallest, the curve $y = \sqrt{3}|x - 1| - 2$ intersects $C$ at points $A$ and $B$, and point $M(1, -4)$. Find the area of $\triangle MAB$.
""",
    """\
In recent years, the emergence of "shared bicycles" has greatly facilitated the "green travel" for citizens. A shared bicycle company "Mobie" plans to invest a total of 1.2 million yuan in two cities, A and B. According to industry regulations, each city must receive an investment of at least 400,000 yuan. Preliminary market research shows that the profit $P$ in city A and the investment $a$ (in units of 10,000 yuan) satisfy $P=3 \sqrt{2a}-6$, and the profit $Q$ in city B and the investment $a$ (in units of 10,000 yuan) satisfy $Q= \frac{1}{4}a+2$. Let the investment in city A be $x$ (in units of 10,000 yuan), and the total profit of the two cities be $f(x)$ (in units of 10,000 yuan). $(1)$ When the investment in city A is 500,000 yuan, calculate the total profit of the company at this time; $(2)$ How should the investments in cities A and B be arranged to maximize the total profit?
""",
    """
From points \( M \) and \( K \), which are 70 km apart, a bus and a cyclist set out towards each other simultaneously. They met after 1 hour and 24 minutes. Continuing at the same speed, the bus arrived at \( K \) and left for the return journey after a 20-minute stop. Find the speeds of the bus and the cyclist if the bus overtook the cyclist 2 hours and 41 minutes after their first meeting.
""",
    """\
Let $ L$ denote the set of all lattice points of the plane (points with integral coordinates). Show that for any three points $ A,B,C$ of $ L$ there is a fourth point $ D,$ different from $ A,B,C,$ such that the interiors of the segments $ AD,BD,CD$ contain no points of $ L.$ Is the statement true if one considers four points of $ L$ instead of three?
""",
    """\
For n real numbers  $a_{1},\, a_{2},\, \ldots\, , a_{n},$  let  $d$  denote the difference between the greatest and smallest of them and  $S = \sum_{i<j}\left |a_i-a_j \right|.$  Prove that \[(n-1)d\le S\le\frac{n^{2}}{4}d\] and find when each equality holds.
""",
]

proof_questions = [
    """\
Given positive integers \(a\) and \(b\) such that \(b > a > 1\), and \(a\) does not divide \(b\), and a given sequence of positive integers \(\{b_n\}_{n=1}^{\infty}\) satisfying \(b_{n+1} \geq 2b_n\) for all positive integers \(n\). Does there always exist a sequence of positive integers \(\{a_n\}_{n=1}^{\infty}\) such that for all positive integers \(n\), \(a_{n+1} - a_n \in \{a, b\}\), and for all positive integers \(m\) and \(l\) (which can be the same), \(a_m + a_l \notin \{b_n\}\) for all \(n\)?
""",
    """\
Let \( f(x) = x^n, x \in D, n \in \mathbf{N}^{+} \). Determine whether \( f(x) \) is a solution to the functional inequality
\[ 
f(x) + f(1-x) > 1 
\]
If so, find the domain \( D \); if not, provide an explanation.
""",
    """\
In a right angled-triangle $ABC$, $\angle{ACB} = 90^o$. Its incircle $O$ meets $BC$, $AC$, $AB$ at $D$,$E$,$F$ respectively. $AD$ cuts $O$ at $P$. If $\angle{BPC} = 90^o$, prove $AE + AP = PD$.
""",
    """\
A(x,y), B(x,y), and C(x,y) are three homogeneous real-coefficient polynomials of x and y with degree 2, 3, and 4 respectively. we know that there is a real-coefficient polinimial R(x,y) such that $B(x,y)^2-4A(x,y)C(x,y)=-R(x,y)^2$. Show that there exist 2 polynomials F(x,y,z) and G(x,y,z) such that $F(x,y,z)^2+G(x,y,z)^2=A(x,y)z^2+B(x,y)z+C(x,y)$ if for any x, y, z real numbers $A(x,y)z^2+B(x,y)z+C(x,y)\ge 0$
""",
    """\
Prove \[\frac{1}{\cos 0^\circ \cos 1^\circ} + \frac{1}{\cos 1^\circ \cos 2^\circ} + \cdots + \frac{1}{\cos 88^\circ \cos 89^\circ} = \frac{\cos 1^\circ}{\sin^2 1^\circ}.\]
""",
]

no_proof_questions = [
    """\
In a $100 \times 25$ rectangular table, each cell is filled with a non-negative real number. The number in the $i$-th row and $j$-th column is denoted by $x_{i, j}$ $(i=1,2,\ldots, 100; j=1,2,\ldots, 25)$ (Table 1). The numbers in each column of Table 1 are then reordered in descending order to create Table 2 such that $x_{1, j}^{\prime} \geq x_{2, j}^{\prime} \geq \cdots \geq x_{100, j}^{\prime}$ $(j=1,2,\ldots, 25)$. Find the smallest natural number $k$ such that if the numbers in Table 1 satisfy $\sum_{j=1}^{25} x_{i, j} \leq 1$ $(i=1,2,\ldots, 100)$, then for $i \geq k$, Table 2 satisfies $\sum_{j=1}^{25} x_{i, j}^{\prime} \leq 1$ $(i=1,2,\ldots, 100)$.
""",
    """\
We are given $2n$ natural numbers
\[1, 1, 2, 2, 3, 3, \ldots, n - 1, n - 1, n, n.\]
Find all $n$ for which these numbers can be arranged in a row such that for each $k \leq n$, there are exactly $k$ numbers between the two numbers $k$.
""",
    """\
Determine all positive integers $n$, $n\ge2$, such that the following statement is true: If $(a_1,a_2,...,a_n)$ is a sequence of positive integers with $a_1+a_2+\cdots+a_n=2n-1$, then there is block of (at least two) consecutive terms in the sequence with their (arithmetic) mean being an integer.
""",
    """\
Turbo the snail sits on a point on a circle with circumference $1$. Given an infinite sequence of positive real numbers $c_1, c_2, c_3, \dots$, Turbo successively crawls distances $c_1, c_2, c_3, \dots$ around the circle, each time choosing to crawl either clockwise or counterclockwise.
Determine the largest constant $C > 0$ with the following property: for every sequence of positive real numbers $c_1, c_2, c_3, \dots$ with $c_i < C$ for all $i$, Turbo can (after studying the sequence) ensure that there is some point on the circle that it will never visit or crawl across.
""",
    """\
For an even integer positive integer $n$ Kevin has a tape of length $4 n$ with marks at $-2 n,-2 n+1, \ldots, 2 n-1,2 n$. He then randomly picks $n$ points in the set $-n,-n+1,-n+2, \ldots, n-1, n$, and places a stone on each of these points. We call a stone 'stuck' if it is on $2 n$ or $-2 n$, or either all the points to the right, or all the points to the left, all contain stones. Then, every minute, Kevin shifts the unstuck stones in the following manner: He picks an unstuck stone uniformly at random and then flips a fair coin. If the coin came up heads, he then moves that stone and every stone in the largest contiguous set containing that stone one point to the left. If the coin came up tails, he moves every stone in that set one point right instead. He repeats until all the stones are stuck. Let $p_{k}$ be the probability that at the end of the process there are exactly $k$ stones in the right half. Evaluate $$\frac{p_{n-1}-p_{n-2}+p_{n-3}-\ldots+p_{3}-p_{2}+p_{1}}{p_{n-1}+p_{n-2}+p_{n-3}+\ldots+p_{3}+p_{2}+p_{1}}$$ in terms of $n$.
""",
    """\
A 0-1 sequence of length $2^k$ is given. Alice can pick a member from the sequence, and reveal it (its place and its value) to Bob. Find the largest number $s$ for which Bob can always pick $s$ members of the sequence, and guess all their values correctly.

Alice and Bob can discuss a strategy before the game with the aim of maximizing the number of correct guesses of Bob. The only information Bob has is the length of the sequence and the member of the sequence picked by Alice.
""" 
]


def formatted_items(problem, filter_type):
    if filter_type == "multiple_choice":
        prompt = f"""\
Given this question: {problem}

Is this a multiple choice question (a question that provides specific options to choose from, typically labeled as A, B, C, D or 1, 2, 3, 4)?

Return only "yes" or "no" without any additional explanation.
"""
    elif filter_type == "proof":
        prompt = f"""\
Given this question: {problem}

Is this a mathematical proof question (a question that asks to prove a statement, theorem, or property...)?

Examples of proof indicators:
- "Prove that..."
- "Show that..."
- "Demonstrate why..."
- "Justify your answer..."
- "Explain why..."
etc.
Here are examples of proof questions:
Example 1: {proof_questions[0]}
Example 2: {proof_questions[1]}
Example 3: {proof_questions[2]}
Example 4: {proof_questions[3]}
Example 5: {proof_questions[4]}
Here are examples of non-proof questions:
Example 1: {no_proof_questions[0]}
Example 2: {no_proof_questions[1]}
Example 3: {no_proof_questions[2]}
Example 4: {no_proof_questions[3]}
Example 5: {no_proof_questions[4]}
Example 6: {no_proof_questions[5]}

Return only "yes" or "no" without any additional explanation.
"""
    elif filter_type == "yes_no":
        prompt = f"""\
Given this question: {problem}

Is this a yes/no question (a question that asks to choose between two options, typically labeled as yes or no)?

Return only "yes" or "no" without any additional explanation.
"""
    elif filter_type == "true_false":
        prompt = f"""\
Given this question: {problem}

Is this a true/false question (a question that asks to choose between two options, typically labeled as true or false)?

Return only "true" or "false" without any additional explanation.
"""
    elif filter_type == "multiple_part":
        prompt = f"""\
Your task is to determine if the given question contains multiple sub-questions, sub-parts, or sub-tasks.
A multi-part question requires separate answers for different components, rather than a single comprehensive answer.
Besides that, if the question is multiple choice and only requires to select one option, it is not a multi-part question.

Here are examples of multi-part questions that require multiple distinct answers:
Example 1: {multiple_parts[0]}
Example 2: {multiple_parts[1]}
Example 3: {multiple_parts[2]}
Example 4: {multiple_parts[3]}
Example 5: {multiple_parts[4]}
Example 6: {multiple_parts[5]}
Example 7: {multiple_parts[6]}
Example 8: {multiple_parts[7]}
Example 9: {multiple_parts[8]}

Here are examples of single-part questions that require only one answer:
Example 1: {no_multiple_parts[0]}
Example 2: {no_multiple_parts[1]}
Example 3: {no_multiple_parts[2]}
Example 4: {no_multiple_parts[3]}
Example 5: {no_multiple_parts[4]}
Example 6: {no_multiple_parts[5]}
Example 7: {no_multiple_parts[6]}
Please analyze this question: {problem}

Does this question contain multiple parts requiring separate answers?
Return only "yes" or "no" without any additional explanation.
"""
    else:
        raise ValueError(f"Invalid type: {filter_type}")

    return [{"role": "user", "content": prompt}]


async def main():
    # Configuration
    model_name = args.model_name
    dataset_name = args.dataset_name
    save_name = args.save_name
    save_folder = args.save_folder
    problem_column_name = args.problem_column_name
    if args.dataset_name_outputs:
        hf_save_dataset_name = args.dataset_name_outputs
    else:
        hf_save_dataset_name = dataset_name
    os.makedirs(save_folder, exist_ok=True)

    batch_size = args.batch_size
    save_interval = args.save_interval

    # Sampling hyperparameters
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

    # prepare data for generation
    items = {}
    if args.multiple_choice:
        items['multiple_choice'] = [formatted_items(row[problem_column_name], "multiple_choice") for _, row in df.iterrows()]
    if args.proof:
        items['proof'] = [formatted_items(row[problem_column_name], "proof") for _, row in df.iterrows()]
    if args.yes_no:
        items['yes_no'] = [formatted_items(row[problem_column_name], "yes_no") for _, row in df.iterrows()]
    if args.true_false:
        items['true_false'] = [formatted_items(row[problem_column_name], "true_false") for _, row in df.iterrows()]
    if args.multiple_part:
        items['multiple_part'] = [formatted_items(row[problem_column_name], "multiple_part") for _, row in df.iterrows()]


    # Process items in batches
    total_items = len(df)
    count = 0
    with SGLangServerManager(model_name, tp=args.tp) as server_handler:
        for idx in tqdm(range(0, total_items, batch_size)):
            print(f"Processing indices {idx}:{idx+batch_size}...")
            for filter_type in items.keys():
                batch_items = items[filter_type][idx:idx+batch_size]
                batch_outputs = await server_handler.get_chat_responses(
                    batch_items,
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
                    if isinstance(response_list, list) and len(response_list) > args.n:
                        print(f"Response List Error: Length > n: {response_list}")
                        response_list = response_list[:args.n]
                    # if processing a single output (this is the usual case), extract the response from the list
                    if args.n == 1:
                        response_list = response_list[0]
                    df.at[idx+i, f"{args.output_model_name}_{filter_type}"] = response_list

            # Save checkpoint
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
    parser.add_argument("--output_model_name", type=str, required=True,
                        help="Name to be prepended to new column names.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_name_outputs", type=str,
                        help="To save the outputs to a different HF dataset, specify here.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--problem_column_name", type=str, default="problem")
    
    # Filters configuration
    parser.add_argument("--multiple_choice", action="store_true")
    parser.add_argument("--proof", action="store_true")
    parser.add_argument("--yes_no", action="store_true")
    parser.add_argument("--true_false", action="store_true")
    parser.add_argument("--multiple_part", action="store_true")

    # Save configuration
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="Save every n batches.")

    # SGLang server configuration
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Total batch size will be args.batch_size * args.n.")

    # LLM Hyperparameters
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=5)
    parser.add_argument("--n", type=int, default=1)

    # dataset slicing
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    asyncio.run(main())

# Example Usage:
# python3 model_based_filters.py --model_name "meta-llama/Meta-Llama-3.1-70B-Instruct" --output_model_name "llama3-70b" --dataset_name "RLAIF/INTERNAL-ONLY-Big-Math-RL-Verified-MC-Rewrites" --dataset_name_outputs "RLAIF/model_filter_testing" --save_folder "outputs" --save_name "model_based_filters" --multiple_choice --proof --yes_no --true_false --multiple_part --tp 8 > model_based_filter_outputs.txt 2>&1