from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_105.prompt as prompt_custom
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
        
        # Validate the solution for correctness
        validation_instruction = f"Validate the following solution: {solution['response']}"
        validation_result = await self.custom(input=validation_instruction, instruction=prompt_custom.VALIDATE_PROMPT)
        
        # Integrate self-ask strategy to enhance output reflection
        self_ask_instruction = f"Given the solution: {solution['response']}, reflect and iterate on your answer for thoroughness."
        self_ask_result = await self.custom(input=self_ask_instruction, instruction=prompt_custom.REFLECT_PROMPT)

        # Validate the solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {self_ask_result['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Use ensemble to improve final answer selection from multiple solutions
        ensemble_result = await self.ensemble(solutions=[solution['response'], validation_result['output'], self_ask_result['response']], problem=problem)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
