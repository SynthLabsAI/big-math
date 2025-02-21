from typing import Optional
import dspy


class MathReformulationSignature(dspy.Signature):
    """You are an AI assistant specializing in transforming multiple-choice math problems into open-ended, solvable questions suitable for an automatic grading system that relies on regex pattern matching within `\boxed{}`. Your task is to reformulate the given problem while adhering to specific guidelines.

**Rules:**
- Never turn problems into proofs, all problems should have an answer that is machine-verifiable
- Reformulated problems should have the same final answer as the original problem (just not in the multiple-choice format)
- It must be reasonably clear from the problem what the student is expected to write inside of \boxed{final_answer}. E.g. "What are its key properties and characteristics?" is almost always a bad reformulation because it is ambiguous what the student should write inside of the `\boxed{}`. "Express your answer in centimeters." is a good answer format specification because it does NOT give away the answer but does specify a clear format.
- Do NOT reformulate problems that are not actually multiple choice or are impossible to reformulate
- If a problem is not multiple choice, return "N/A" for the reformulated problem and other "N/A" fields

**Follow these steps:**

1. **Analyze the original problem:**
    - Identify the core mathematical concept.
    - Note any crucial information, including numerical values, equations, and key terms.
    - Determine the structure of the problem (e.g., scenario-based, direct question, multi-step).
    - Preserve all mathematical notations, symbols, and formatting as they appear in the original problem.

2. **Remove multiple-choice options:**
    - Eliminate all answer choices and their labels (A, B, C, etc.).
    - If any options contain information essential to solving the problem, incorporate that information into the main problem statement.

3. **Rephrase the question:**
    - Transform the question into an open-ended format that requires a direct numerical or algebraic answer.
    - Ensure the rephrased question is clear, unambiguous, and uses language appropriate for the student's level.

4. **Maintain problem integrity:**
    - Keep all original numerical values, equations, figures, and key terms intact.
    - Preserve any scenarios, dialogues, or conditional information crucial for solving the problem.
    - Do not introduce new information, alter units of measurement, or change the mathematical intent of the problem.
    - If the problem references diagrams or figures, ensure that any necessary descriptions are included.

5. **Specify the answer format:**
    - Instruct the student to provide their answer using the `\boxed{}` format.
    - Do not include placeholders like `\boxed{N}`; instead, guide the student to input their calculated answer within the boxed format.
    - For example, "Provide your answer in the form `\boxed{\text{your answer here}}`."

6. **Final check:**
    - Ensure the reformulated problem contains all necessary information for independent solving.
    - Verify that the problem hasn't become easier or harder than the original.
    - Check for any common errors, such as unit inconsistencies or typographical mistakes.
    - Confirm that no hints, solution methods, or additional explanations have been inadvertently added.

7. **Is actually multiple choice:**
    - Some problems are not actually multiple choice and do NOT actually need to be reformulated.
    - If the problem is **NOT** multiple choice, do NOT reformulate it! Note here if it is not multiple choice and return "N/A" for the reformulated problem.

**Before providing the final reformulated problem, please create a `reformulation_process` dictionary (for internal use; do not include this dictionary or its content in the final problem). The dictionary should have the following exact keys:**

- `"core_mathematical_concept"`: Summarize the core mathematical concept.
- `"key_information_extraction"`: List key information (numerical values, equations, terms).
- `"problem_structure_analysis"`: Identify the problem structure.
- `"multiple_choice_removal_strategy"`: Plan how to remove multiple-choice options.
- `"rephrasing_approach"`: Outline the rephrasing strategy.
- `"problem_integrity_preservation"`: Note how to maintain problem integrity.
- `"answer_format_specification"`: Specify the answer format.
- `"is_multiple_choice"`: Whether the problem is actually multiple choice.

**This structured approach will help ensure accuracy and completeness in your final output.**

**After your reformulation process, present the reformulated problem as plain text, making sure it meets all the criteria outlined above. Do not include the `<reformulation_process>` tags or any of their content in the final problem presented to the student.**

**Examples:**
```
<example>
<problem>
Three squares with the same center and corresponding parallel sides are drawn. The distance between the sides of successive squares is 3 units, and the side length of the largest square is 22 units. What is the perimeter of the smallest square?
(A) 40
(B) 64
(C) 100
(D) 10
(E) 20
</problem>

<reformulation_process>
{
    "core_mathematical_concept": "Geometry - concentric squares and their dimensions",
    "key_information_extraction": [
        "Three concentric squares",
        "Distance between sides of successive squares: 3 units",
        "Side length of the largest square: 22 units",
        "Need to find the perimeter of the smallest square"
    ],
    "problem_structure_analysis": "Direct question based on given information",
    "multiple_choice_removal_strategy": [
        "Eliminate all answer choices and their labels",
        "No essential information is lost by removing the options"
    ],
    "rephrasing_approach": [
        "Keep the description of the squares intact",
        "Rephrase the question to ask for the perimeter in an open-ended format",
        "Specify units and number format requirements"
    ],
    "problem_integrity_preservation": [
        "Preserve all original numerical values and key terms",
        "Do not add any new information or hints"
    ],
    "answer_format_specification": [
        "Answer should be expressed in units",
        "Answer should be given as a whole number",
        "No decimal places required based on original options"
    ],
    "is_multiple_choice": true
}
</reformulation_process>

<reasoning>
Let me think about this step by step:

1. Analyzing the multiple choice aspect:
   - The options are all whole numbers: 40, 64, 100, 10, 20
   - No units are specified in the options but units are used in the problem
   - The options suggest an exact answer is expected, not an approximation

2. Looking at the essential information:
   - The geometric relationships are clearly stated
   - All measurements are given in units
   - No complex transformations or unit conversions needed

3. What needs to be preserved:
   - The concentric squares description
   - The 3-unit distance between squares
   - The 22-unit largest square side length
   - The focus on perimeter rather than area

4. What needs to be clarified:
   - The answer should include units
   - A whole number is expected
   - No decimal places needed

5. How to make it open-ended:
   - Remove all multiple choice options
   - Keep the geometric scenario exactly as is
   - Add specific answer format requirements
   - Maintain the same level of difficulty

6. Final considerations:
   - The reformulated question should be solvable without reference to the original options
   - The specifications should ensure an unambiguous answer format
   - The mathematical challenge should remain unchanged
</reasoning>

<reformulated_problem>
Three squares with the same center and corresponding parallel sides are drawn. The distance between the sides of each successive square is 3 units, and the side length of the largest square is 22 units. Find the perimeter of the smallest square. Express your answer in units as a whole number.
</reformulated_problem>
</example>
```

```
<example>
<problem>
The teacher picked 9 out of 13 cards labeled with numbers 1 to 13 and stuck them on the foreheads of 9 students. Each student can see the numbers on the other 8 students' foreheads but not their own. The teacher asked, "Those who know the number of divisors of their own number, please raise your hand." Two students raised their hands. After they put their hands down, three students had the following conversation.
- A: "I know my number now."
- B: "I still don't know what my number is, but I have figured out whether it is odd or even."
- C: "My number is 2 less than B's number and 1 more than A's number."
What is the sum of the numbers on the four cards that were not picked?
</problem>

<reformulation_process>
{
"core_mathematical_concept": "Number theory and logical deduction with combinatorics",
"key_information_extraction": [
    "13 cards numbered from 1 to 13",
    "9 cards picked and assigned to students",
    "Students can see others' numbers but not their own",
    "Two students know their number based on number of divisors",
    "Conversation between students A, B, and C revealing relationships",
    "Need to find sum of the 4 unpicked cards"
],
"problem_structure_analysis": "Complex logical deduction problem combining number theory with information from multiple stages",
"multiple_choice_removal_strategy": "Not applicable as this is not a multiple choice question",
"rephrasing_approach": "Not needed as problem is already in appropriate format",
"problem_integrity_preservation": "No modifications needed",
"answer_format_specification": "Answer should be expressed as a single integer (sum of the unpicked cards)",
"is_multiple_choice": false
}
</reformulation_process>

<reasoning>
Let me think about this step by step:

1. First, assess if reformulation is needed:
   - This is not a multiple choice question
   - It already asks for a specific numerical answer
   - The format is clear and unambiguous

2. Analyzing the problem structure:
   - It's a complex logic puzzle
   - All information provided is necessary
   - The sequence of events matters
   - The dialogue provides crucial information

3. Consider what might need clarification:
   - Answer format is implicitly clear (a sum)
   - No unit specification needed
   - No decimal places involved (working with integers 1-13)
   - No ambiguity in what constitutes an answer

4. Check for potential improvements:
   - Current format is optimal for the problem type
   - Adding structure might make it harder to follow
   - The narrative flow is important for understanding

5. Final decision:
   - No reformulation needed
   - Problem is already in ideal open-ended format
   - Answer requirements are clear (single integer sum)
</reasoning>

<reformulated_problem>
N/A
</reformulated_problem>
</example>
```

```
<example>
<problem>
In the cube shown, \( L \) and \( K \) are midpoints of adjacent edges \( AD \) and \( AB \), respectively. The perpendicular distance from vertex \( F \) to the line segment \( LK \) is 10 units. Which of the following measurements is closest to the volume of the cube after subtracting 3 units?

(A) 200 in³
(B) 500 cm³
(C) 0.01039 m³
(D) Tie
</problem>

<reformulation_process>
{
    "core_mathematical_concept": "3D geometry involving cube volume calculation",

    "key_information_extraction": [
        "Points L and K are midpoints of adjacent edges AD and AB",
        "Perpendicular distance from F to LK is 10 units",
        "Need to subtract 3 units from final volume"
    ],

    "problem_structure_analysis": "Geometry problem requiring volume calculation",

    "multiple_choice_removal_strategy": [
        "Remove all comparisons",
        "Ask for direct volume calculation"
    ],

    "rephrasing_approach": [
        "Keep geometric relationships intact",
        "Request exact volume calculation",
        "Specify units and precision"
    ],

    "problem_integrity_preservation": [
        "Maintain all numerical values and geometric relationships",
        "Remove comparative aspect entirely"
    ],

    "answer_format_specification": [
        "Answer should be expressed in cubic meters (m³)",
        "Round to 5 decimal places",
        "Include units in answer"
    ],

    "is_multiple_choice": true
}
</reformulation_process>
<reasoning>
Let me think about this step by step:

1. First, I need to identify what makes this multiple choice:

   - The core task is actually calculating a specific volume
   - The multiple choice aspect is just about comparing to different unit measurements
   - The comparison isn't fundamental to the mathematical concept being tested

2. Looking at the original answer choices:

   - They use different units (in³, cm³, m³)
   - The m³ value has 5 decimal places (0.01039)
   - This suggests m³ with 5 decimal places is an appropriate precision level

3. How to remove the multiple choice aspect:

   - Instead of asking which measurement is closest
   - Ask for the exact calculation
   - Need to specify one consistent unit to avoid ambiguity

4. What to preserve:

   - All geometric relationships
   - The 3-unit subtraction
   - The original scenario and values

5. What needs to be added:

   - Specific unit requirement (m³)
   - Decimal place requirement (5 places)
   - Clear instruction about including units

6. The reformulated question should:

   - Be completely open-ended
   - Have a unique, calculable answer
   - Remove all traces of comparison
   - Maintain the same difficulty level

</reasoning>
<reformulated_problem>
In the cube shown, \( L \) and \( K \) are midpoints of adjacent edges \( AD \) and \( AB \), respectively. The perpendicular distance from vertex \( F \) to the line segment \( LK \) is 10 units. Calculate the volume of the cube after subtracting 3 cubic units. Express your answer in cubic meters (m³), rounded to 5 decimal places.
</reformulated_problem>
</example>
```
    """

    original_problem: str = dspy.InputField(
        desc="""The original multiple-choice math problem text, including all options and formatting."""
    )

    correct_answer: str = dspy.InputField(
        desc="The correct answer to the original problem."
    )

    reformulation_process: dict = dspy.OutputField(
        desc="""Structured analysis of the problem reformulation process containing the following exact keys:
        {
            "core_mathematical_concept": "Identify & summarize state the main mathematical concept",
            "key_information_extraction": "List of numerical values, equations, and key terms",
            "problem_structure_analysis": "Description of problem type and structure",
            "multiple_choice_removal_strategy": "How to handle removing MC options",
            "rephrasing_approach": "Strategy for rephrasing as open-ended",
            "problem_integrity_preservation": "How to maintain original difficulty and context",
            "answer_format_specification": "How the answer should be formatted with \\boxed{}"
        }""",
        prefix="<reformulation_process>\n\n{"
    )

    reasoning: str = dspy.OutputField(
        desc="""Think step by step about how to reformulate the problem while adhering to the guidelines provided. Or, if the problem is NOT a multiple choice problem, justify why it is not.""",
        prefix="<reasoning>\n\n"
    )

    reformulated_problem: str = dspy.OutputField(
        desc=r"""The reformulated open-ended problem that:
        - Preserves all mathematical notations and symbols
        - Maintains the original difficulty level
        - Specifies, and if necessary has clear instructions, for the expected answer format. The student should not be confused about the type of answer that should be presented. (E.g. round your answer to the nearest hundredth, or express your answer as a percentage, list your answer as a coordinate, etc.). If the answer cannot be easily expressed in boxed latex format (for auto-grading), then the problem was not well formulated!
        - Contains no multiple-choice options
        - Does not add any hints or solution methods
        - Retains all necessary context to solve the problem
        (If the problem is not a multiple choice problem or cannot be reformulated, then just return N/A)""",
        prefix="<reformulated_problem>\n\n"
    )

    solution: str = dspy.OutputField(
        desc="""The solution to the reformulated problem. What the student should write inside of the `\boxed{}`.""",
        prefix="<solution>\n\n"
    )


class ReformulationJudgeSignature(dspy.Signature):
    """You are a IMO judge, specialized in evaluating the quality of IMO math problem reformulations. Your task is to critically analyze whether a reformulated problem truly meets the criteria for being open-ended, challenging, and maintains the original problem's integrity.

**Important:**
- Problems you review are being transformed from multiple choice problems into open-ended, solvable questions suitable for an automatic grading system that relies on regex pattern matching within `\boxed{}`. It is VERY important that even though the problem is no longer multiple choice, it still has a single, definite answer.
- It must be reasonably clear from the problem what the student is expected to write inside of \boxed{final_answer}. E.g. "What are its key properties and characteristics?" is almost always a bad reformulation because it is ambiguous what the student should write inside of the `\boxed{}`. "Express your answer in centimeters." is a good answer format specification because it does NOT give away the answer but does specify a clear format.

Key aspects to evaluate:

1. Hidden Multiple Choice
    - Check if the reformulation still effectively presents multiple choice options by:
        - Embedding a limited/fixed set of choices (e.g. 4 or 5 options) within the problem text that limits the answer choices to one of those options
        - Asking to compare with specific values
        - Limiting answers to specific options
    - Flag any reformulations that are just disguised multiple choice questions

Example:

<problem>
For real numbers $t \neq 0,$ the point \[(x,y) = \left( \frac{t + 1}{t}, \frac{t - 1}{t} \right)\]is plotted. All the plotted points lie on what kind of curve? (A) Line (B) Circle (C) Parabola (D) Ellipse (E) Hyperbola Enter the letter of the correct option.
</problem>

Reformulated failed example (all multiple choice aspect still present):
<bad_reformulated_problem>
For real numbers $t \neq 0,$ the point \[(x,y) = \left( \frac{t + 1}{t}, \frac{t - 1}{t} \right)\]is plotted. What type of curve do all the plotted points lie on? Provide your answer as a specific curve type (e.g., line, circle, parabola, ellipse, hyperbola).
</bad_reformulated_problem>

Reformulated successful example (multiple choice aspect removed):
<reformulated_problem>
For real numbers \( t \neq 0 \), the point
\[
(x, y) = \left( \frac{t + 1}{t}, \frac{t - 1}{t} \right)
\]
is plotted. Determine the type of coordinate geometry curve on which all the plotted points lie.
</reformulated_problem>

2. Mathematical Integrity
    - Verify that the mathematical difficulty remains unchanged
    - Ensure no accidental hints or simplifications were introduced
    - Check that all necessary information was preserved
    - Confirm no extraneous information was added

Example 1:
This problem was not actually multiple choice, but it does _imply_ options that are clearly missing/omitted.
<problem>
A resident wants to renovate their house and buys several strips of wood, each with a length of 0.7 meters and 0.8 meters. By connecting some of these wood strips, many different lengths of wood can be obtained. For example, \(0.7 + 0.7 = 1.4\) meters, \(0.7 + 0.8 = 1.5\) meters, etc. From the options below, what length of wood strip cannot be obtained by connecting these wood strips?
</problem>


Example 2:
<problem>
$\triangle ABC$ is inscribed in a semicircle of radius $r$ so that its base $AB$ coincides with diameter $AB$. Point $C$ does not coincide with either $A$ or $B$. Let $s=AC+BC$. Then, for all permissible positions of $C$: $\textbf{(A)}\ s^2\le8r^2\qquad \textbf{(B)}\ s^2=8r^2 \qquad \textbf{(C)}\ s^2 \ge 8r^2 \qquad\\ \textbf{(D)}\ s^2\le4r^2 \qquad \textbf{(E)}\ s^2=4r^2$
</problem>

The reformulation is flawed because it prematurely focuses on a single inequality (\(s^2 \leq 8r^2\)) rather than inviting exploration of the entire range of \(s^2\). It biases the solver, limits generality, and reduces the problem's open-ended nature.
<bad_reformulated_problem>
$\triangle ABC$ is inscribed in a semicircle of radius $r$ so that its base $AB$ coincides with diameter $AB$. Point $C$ does not coincide with either $A$ or $B$. Let $s=AC+BC$. Prove or disprove the inequality $s^2\le8r^2$ for all permissible positions of $C$.
</bad_reformulated_problem>

An open-ended revision that requires the same analysis and leads to the same conclusion
<reformulated_problem>
Let triangle \( ABC \) be inscribed in a semicircle of radius \( r \), with its base \( AB \) coinciding with the diameter \( AB \). Point \( C \) lies on the semicircle but does not coincide with \( A \) or \( B \). Let \( s = AC + BC \). Determine the maximum possible value of \( s^2 \) in terms of \( r \), and prove that \( s^2 \leq 8r^2 \) for all permissible positions of \( C \).
</reformulated_problem>

Example 3:
<problem>
For real numbers $t,$ the point \[(x,y) = \left( \frac{1 - t^2}{1 + t^2}, \frac{2t}{1 + t^2} \right)\]is plotted. All the plotted points lie on what kind of curve? (A) Line (B) Circle (C) Parabola (D) Ellipse (E) Hyperbola Enter the letter of the correct option.
</problem>

Although it removes the multiple-choice format, it does not truly open the problem to exploration. The answer asks for a description, which introduces too much variability in responses
<bad_reformulated_problem>
For real numbers $t,$ the point \[(x,y) = \left( \frac{1 - t^2}{1 + t^2}, \frac{2t}{1 + t^2} \right)\]is plotted. What kind of curve do all the plotted points lie on? Provide a brief description of the curve.
</bad_reformulated_problem>

This reformulation is clear, concise, and ensures consistent answers while remaining open-ended for exploration.
<reformulated_problem>
For real numbers \( t \), consider the point

\[
(x, y) = \left( \frac{1 - t^2}{1 + t^2},\ \frac{2t}{1 + t^2} \right).
\]

Determine the type of curve on which all such points \((x, y)\) lie. State your answer as the name of the curve.
</reformulated_problem>

Example 4:

<problem>
Define * as an operation on ordered pairs of real numbers, such that $(a, b) *(c, d)=$ $(a c+b d, a d+b c)$. If $(a, b) *(x, y)=(a, b)$, then which of the following is $(x, y)$? (A) $(0,0)$. (B) $(0,1)$. (C) $(1,0)$. (D) $(1,1)$. (E) $(1,-1)$.
</problem>

Ambiguous about whether the equation holds for all \((a, b)\) or specific values, lacks context to ensure a unique solution, and removes guidance provided by the multiple-choice format. Becomes unbounded.
<bad_reformulated_problem>
Define * as an operation on ordered pairs of real numbers, such that $(a, b) *(c, d)=$ $(a c+b d, a d+b c)$. If $(a, b) *(x, y)=(a, b)$, then calculate the ordered pair $(x, y)$. Express your answer in the format $(x, y)$, including parentheses and a comma.
</bad_reformulated_problem>

Ideal because it clarifies the scope (applies to all \((a, b)\)), ensures a unique solution, and balances open-ended exploration with clear grading criteria
<reformulated_problem>
Define \( * \) as an operation on ordered pairs of real numbers, such that
\[
(a, b) * (c, d) = (ac + bd, ad + bc).
\]
Find the ordered pair \((x, y)\) that satisfies
\[
(a, b) * (x, y) = (a, b)
\]
for **all** real numbers \(a\) and \(b\). Express your answer as \((x, y)\).
</reformulated_problem>

3. Answer Format Clarity
    - Evaluate if the answer format specification is:
        - Clear and unambiguous
        - Appropriate for the mathematical concept
        - Not overly unbounded or restrictive in a way that creates an open-ended problem out of it
    - Check if unit/precision requirements make sense for the problem

4. Problem Independence
    - Verify the reformulated problem can stand alone
    - Ensure it doesn't rely on knowledge of the original options
    - Check that answer requirements aren't derived solely from original choices

Example:
<problem>
Which of the following is a root of the equation \( x^2 - x - 6 = 0 \)?
(A) \( -3 \) (B) \( -2 \) (C) \( 2 \) (D) \( 3 \) (E) \( 6 \)
</problem>

Relies on original options, failing to stand independently and limiting exploration.
<bad_reformulated_problem>
Find a root of the equation \( x^2 - x - 6 = 0 \). Your answer must be one of the following: \( -3, -2, 2, 3, 6 \).
</bad_reformulated_problem>

Ideal because it ensures the solver identifies all roots without being constrained by the original options.
<reformulated_problem>
Solve the quadratic equation \( x^2 - x - 6 = 0 \) and find all real roots. Provide your answers in increasing order.
</reformulated_problem>

Remember: A truly open-ended reformulation should allow for calculation and expression of the answer
without any reference to or knowledge of the original multiple choice options."""

    original_problem: str = dspy.InputField(
        desc="The original multiple-choice math problem text, including all options and formatting."
    )
    correct_answer: str = dspy.InputField(
        desc="The correct answer to the original problem."
    )

    reformulated_problem: str = dspy.InputField(
        desc="The reformulated open-ended version of the problem that needs to be evaluated."
    )

    issues: list[str] = dspy.OutputField(
        desc="List of specific issues found with the reformulation, if any.",
        prefix="<issues>\n"
    )

    suggestions: list[str] = dspy.OutputField(
        desc="Specific recommendations for improving the reformulation.",
        prefix="<suggestions>\n"
    )

    reasoning: str = dspy.OutputField(
        desc="Step-by-step explanation of the evaluation process and rationale for the verdict.",
        prefix="<reasoning>\n"
    )

    verdict: str = dspy.OutputField(
        desc="Either 'PASS' or 'FAIL' based on comprehensive evaluation of the reformulation.",
        prefix="<verdict>\n"
    )

    corrected_version: Optional[str] = dspy.OutputField(
        desc="An improved version of the reformulation if issues were found. Should be None if verdict is PASS.",
        prefix="<corrected_version>\n"
    )