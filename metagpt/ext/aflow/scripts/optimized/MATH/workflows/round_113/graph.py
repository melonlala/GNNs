from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_113.prompt as prompt_custom
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
        multiple_solutions = await self.custom(input=problem, instruction=prompt_custom.GENERATE_MULTIPLE_PROMPT)
        
        # Validate the solutions for correctness
        validation_instructions = [f"Validate the following solution: {sol['response']}" for sol in multiple_solutions['responses']]
        validation_results = await asyncio.gather(*(self.custom(input=inst, instruction=prompt_custom.VALIDATE_PROMPT) for inst in validation_instructions))
        
        validated_solutions = [val['response'] for val in validation_results]
        
        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=validated_solutions, problem=problem)
        
        # Self-ask to reflect on the ensemble result
        reflection_instruction = f"Reflect on the ensemble result: {ensemble_result['response']} and provide an explanation."
        reflection_result = await self.custom(input=reflection_instruction, instruction=prompt_custom.REFLECT_PROMPT)
        
        # Return the final solution and cost
        return reflection_result['response'], self.llm.cost_manager.total_cost
