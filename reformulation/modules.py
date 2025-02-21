import dspy
from loguru import logger

from signatures import MathReformulationSignature, ReformulationJudgeSignature


class MathProblemReformulator(dspy.Module):
    """
    Reformulates multiple-choice math problems into open-ended format.

    Methods
    -------
    __init__():
        Initializes the MathProblemReformulator with prediction models for reformulation and judgment.
    
    forward(problem: str, correct_answer: str) -> dspy.Prediction:
        Reformulates a given multiple-choice math problem and judges the reformulation.
        
        Parameters:
        problem (str): The original multiple-choice math problem.
        correct_answer (str): The correct answer to the original problem.
        
        Returns:
        dspy.Prediction: The prediction result containing the reformulated problem, validation details, and judgment results.
    
    _validate(reformulation_process: dict, reformulated_problem: str) -> tuple[bool, dict]:
        Validates the reformulation output.
        
        Parameters:
        reformulation_process (dict): The process details of the reformulation.
        reformulated_problem (str): The reformulated math problem.
        
        Returns:
        tuple[bool, dict]: A tuple containing a boolean indicating if the validation passed and a dictionary with validation details.
    """
    """Reformulates multiple-choice math problems into open-ended format"""

    def __init__(self):
        super().__init__()
        self.reformulate = dspy.Predict(MathReformulationSignature)
        self.judge = dspy.Predict(ReformulationJudgeSignature)

    def forward(self, problem: str, correct_answer: str):
        # First pass - reformulate the problem
        result = self.reformulate(original_problem=problem, correct_answer=correct_answer)

        # Skip judge if problem wasn't actually multiple choice
        if result.reformulated_problem.strip() == "N/A":
            # Return early with minimal response
            return dspy.Prediction(
                reformulation_process=result.reformulation_process,
                reformulated_problem="N/A",
                validation_result=True,  # Consider it valid since it's properly identified
                validation_details={"not_multiple_choice": True},
                reasoning=result.reasoning,
                reformulation_solution=result.solution,
                judge_corrected_reformulated_problem=None,
                judge_verdict=None,
                judge_issues=None,
                judge_suggestions=None,
                judge_reasoning=None
            )

        # Second pass - judge the reformulation
        judgment = self.judge(
            original_problem=problem,
            correct_answer=correct_answer,
            reformulated_problem=result.reformulated_problem
        )

        # Run basic validation
        is_valid, validation_details = self._validate(
            result.reformulation_process,
            result.reformulated_problem
        )

        # Add corrected_reformulated_problem without overwriting
        corrected_reformulation = judgment.corrected_version if judgment.corrected_version else None

        return dspy.Prediction(
            reformulation_process=result.reformulation_process,
            reformulated_problem=result.reformulated_problem,
            validation_result=is_valid,
            validation_details=validation_details,
            reasoning=result.reasoning,
            reformulation_solution=result.solution,
            judge_corrected_reformulated_problem=corrected_reformulation,
            judge_verdict=judgment.verdict,
            judge_issues=judgment.issues,
            judge_suggestions=judgment.suggestions,
            judge_reasoning=judgment.reasoning
        )

    def _validate(self, reformulation_process: dict, reformulated_problem: str) -> tuple[bool, dict]:
        """Validate reformulation output"""
        # expected keys in snake_case to match model output
        expected_keys = {
            'core_mathematical_concept',
            'key_information_extraction',
            'problem_structure_analysis',
            'multiple_choice_removal_strategy',
            'rephrasing_approach',
            'problem_integrity_preservation',
            'answer_format_specification',
            'is_multiple_choice'
        }

        # Validate reformulation process keys
        provided_keys = set(reformulation_process.keys())
        missing_keys = expected_keys - provided_keys
        extra_keys = provided_keys - expected_keys

        # Log any key mismatches for debugging
        if missing_keys:
            logger.debug(f"Missing keys: {missing_keys}")
        if extra_keys:
            logger.debug(f"Unexpected extra keys: {extra_keys}")

        # # Check if reformulated problem contains boxed instruction
        # has_box_instruction = "\\boxed{" in reformulated_problem

        # Check if multiple choice options were removed
        no_mc_options = not any(x in reformulated_problem for x in ["(A)", "(B)", "(C)", "(D)", "(E)"])

        process_complete = not missing_keys and not extra_keys

        validation_details = {
            # "has_box_instruction": has_box_instruction,
            "no_mc_options": no_mc_options,
            "process_complete": process_complete
        }

        return all(
            [
                # has_box_instruction,
                no_mc_options,
                process_complete
            ]
        ), validation_details