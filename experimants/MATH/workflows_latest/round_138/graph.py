from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_138.prompt as prompt_custom
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
        
        # Self-ask to enhance the generated solution
        self_ask_instruction = f"Reflect on the following solution and question its correctness: {solution['response']}"
        self_ask_result = await self.custom(input=self_ask_instruction, instruction=prompt_custom.SELF_ASK_PROMPT)
        
        # Validate the solution for correctness using combined feedback
        validation_instruction = f"Validate and review the following solution: {solution['response']} {self_ask_result['response']}"
        validation_result = await self.custom(input=validation_instruction, instruction=prompt_custom.VALIDATE_PROMPT)
        
        # Validate the solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {validation_result['response']}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=[solution['response'], validation_result['output']], problem=problem)
        
        # Return the final solution and cost
        return ensemble_result['response'], self.llm.cost_manager.total_cost
