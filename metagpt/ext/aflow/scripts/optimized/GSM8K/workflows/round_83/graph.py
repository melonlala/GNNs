from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_83.prompt as prompt_custom
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
        self.ensemble = operator.ScEnsemble(self.llm)
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Initial validation of the problem statement
        validation_response = await self.custom(input=problem, instruction=prompt_custom.VALIDATE_PROBLEM_PROMPT)
        if validation_response['response'] != "Valid":
            return "Error: Invalid problem statement.", self.llm.cost_manager.total_cost
            
        # Self-ask mechanism to refine the problem statement
        refined_problem = await self.custom(input=validation_response['response'], instruction=prompt_custom.REFINE_PROMPT)
        
        # Generate multiple initial solutions
        initial_solutions = await self.custom(input=refined_problem['response'], instruction=prompt_custom.MULTI_SOLUTIONS_PROMPT)
        
        # Compare and summarize the initial solutions
        solution_comparison = await self.custom(input=f"Compare the following solutions: {initial_solutions['response']}", instruction=prompt_custom.COMPARE_SOLUTIONS_PROMPT)

        # Review the solution comparisons using Programmer for detailed analysis
        review_solutions = await self.programmer(problem=refined_problem['response'], analysis=solution_comparison['response'])
        
        # Validate the solutions using ScEnsemble
        ensemble_solution = await self.ensemble(solutions=[review_solutions['output'], initial_solutions['response']], problem=refined_problem['response'])
        
        # Final review and validation of the ensemble solution
        final_validation = await self.programmer(problem=refined_problem['response'], analysis=ensemble_solution['response'])
        
        # Return the validated output
        return final_validation['output'], self.llm.cost_manager.total_cost
