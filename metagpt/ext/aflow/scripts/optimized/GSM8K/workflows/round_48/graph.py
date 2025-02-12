from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_48.prompt as prompt_custom
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
        # Generate multiple initial solutions
        initial_solutions = await self.custom(input=problem, instruction=prompt_custom.MULTI_SOLUTION_PROMPT)
        
        # Review the initial solutions for validity
        review_solutions = [await self.programmer(problem=problem, analysis=sol['response']) for sol in initial_solutions['responses']]
        
        # Validate solutions multiple times through ensemble
        validated_solutions = []
        for solution in review_solutions:
            ensemble_solution = await self.ensemble(solutions=[solution['output']], problem=problem)
            validated_solutions.append(ensemble_solution['response'])
        
        # Final comprehensive review for correctness
        final_review = await self.custom(input=problem + f" Validate these solutions: {validated_solutions}", instruction=prompt_custom.VALIDATION_PROMPT)
        
        # Final validation step to ensure correctness
        if final_review['response'] == expected_answer:  # Assuming expected_answer is defined
            return final_review['response'], self.llm.cost_manager.total_cost
        else:
            return "Validation failed", self.llm.cost_manager.total_cost
