from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_80.prompt as prompt_custom
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
        # Refine the problem statement before generating solutions
        refined_problem = await self.custom(input="Refine this problem statement: " + problem, instruction="")
        
        # Generate multiple initial solutions using Ensemble
        initial_solutions = await self.ensemble(solutions=[await self.custom(input=refined_problem['response'], instruction=prompt_custom.DETAILED_CALCULATION_PROMPT) for _ in range(3)], problem=refined_problem['response'])
        
        # Aggregate and filter the best solutions based on frequency
        promising_solutions = initial_solutions['response'][:2]  # Take top 2 based on some criteria
        
        # Validate the promising solutions
        validation = await self.custom(input="Evaluate the accuracy of the following solutions: " + promising_solutions, instruction=prompt_custom.VALIDATE_SOLUTION_PROMPT)
        
        if validation['response'] != 'confident':
            # Loop to refine if validation is not confident
            refined_solution = await self.custom(input=promising_solutions, instruction="Refine this solution: ")
            validation = await self.custom(input="Validate the refined solution: " + refined_solution['response'], instruction=prompt_custom.VALIDATE_SOLUTION_PROMPT)

        # Review the validated solution using Programmer for detailed analysis
        review_solution = await self.programmer(problem=refined_problem['response'], analysis=validation['response'])
        
        return review_solution['output'], self.llm.cost_manager.total_cost
