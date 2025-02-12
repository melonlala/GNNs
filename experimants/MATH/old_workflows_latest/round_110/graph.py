from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_110.prompt as prompt_custom
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
        # Generate multiple initial solutions
        candidate_solutions = await self.custom(input=problem, instruction=prompt_custom.GENERATE_MULTIPLE_SOLUTIONS_PROMPT)
        
        # Reflect on the generated solutions for iterative refinement
        reflection_instruction = f"Reflect on these solutions and refine if necessary: {candidate_solutions['response']}"
        refined_solutions = await self.custom(input=reflection_instruction, instruction=prompt_custom.SELF_ASK_PROMPT)
        
        # Validate the refined solutions for correctness
        validation_instruction = f"Validate the following solutions: {refined_solutions['response']}"
        validation_result = await self.custom(input=validation_instruction, instruction=prompt_custom.VALIDATE_PROMPT)
        
        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=[refined_solutions['response'], validation_result['response']], problem=problem)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
