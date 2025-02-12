from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_150.prompt as prompt_custom
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
        initial_solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Generate additional candidate solutions
        multi_solution_instructions = f"Provide alternative solutions for the following problem: {problem}"
        candidate_solutions = await self.custom(input=multi_solution_instructions, instruction=prompt_custom.MULTI_SOLVE_PROMPT)
        
        # Combine the original and additional solutions for review
        combined_solutions = [initial_solution['response']] + candidate_solutions['responses']
        
        # Review and aggregate results based on majority voting
        review_instruction = f"Aggregate the following solutions based on correctness and consistency: {combined_solutions}"
        aggregated_result = await self.custom(input=review_instruction, instruction=prompt_custom.AGGREGATE_REVIEW_PROMPT)
        
        # Use ensemble to select the best final answer
        ensemble_result = await self.ensemble(solutions=[aggregated_result['response']], problem=problem)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
