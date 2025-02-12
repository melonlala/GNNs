from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_122.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.llm.cost_manager = CostManager()
        self.custom = operator.Custom(self.llm)
        self.programmer = operator.Programmer(self.llm)
        self.ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate initial solution
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Validate the initial solution
        validation_instruction = f"Validate if the following solution is correct for the problem: {problem}. Solution: {solution['response']}"
        validation_result = await self.custom(input=validation_instruction, instruction=prompt_custom.VALIDATE_PROMPT)
        
        # Self-ask for alternative approaches
        self_ask_instruction = f"Provide alternative methods or techniques to solve the problem: {problem}"
        self_ask_result = await self.custom(input=self_ask_instruction, instruction=prompt_custom.SELF_ASK_PROMPT)
        
        # Review the solution for correctness and provide suggestions for improvement
        review_instruction = f"Review the following solution for correctness and provide suggestions for improvement. Initial solution: {solution['response']}. Validation: {validation_result['response']}. Additional insights: {self_ask_result['response']}"
        review_result = await self.custom(input=review_instruction, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Summarize the retrieved insights before final decision
        summary_instruction = f"Summarize the following insights: {solution['response']}, {validation_result['response']}, {self_ask_result['response']}, {review_result['response']}"
        summary_result = await self.custom(input=summary_instruction, instruction=prompt_custom.SUMMARY_PROMPT)

        # Validate the solution using Python code with enhanced logic
        validation_code = f"def validate_solution(solution):\n    return solution == 'expected_output'\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=[solution['response'], validation_result['output']], problem=problem)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
