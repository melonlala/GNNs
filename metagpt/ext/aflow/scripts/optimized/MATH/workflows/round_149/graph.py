from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MATH.workflows.round_149.prompt as prompt_custom
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
        
        # Multi-Review the solution for correctness
        multi_review_instruction = f"Review the following solution from multiple perspectives for correctness: {solution['response']}"
        review_result1 = await self.custom(input=multi_review_instruction, instruction=prompt_custom.REVIEW_PROMPT)
        review_result2 = await self.custom(input=multi_review_instruction, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Combine review results for validation
        combined_review = f"Aggregated insights: {review_result1['response']} | {review_result2['response']}"
        
        # Validate the solution using Python code
        validation_code = f"def validate_solution():\n    # Your validation logic here\n    return {combined_review}\n"
        validation_result = await self.programmer(problem=validation_code)
        
        # Use ensemble to improve the final answer selection
        ensemble_result = await self.ensemble(solutions=[solution['response'], validation_result['output']], problem=problem)
        
        # Extract final answer with enhanced functionality to capture boxed outputs
        final_answer = extract_model_answer(ensemble_result['response'])
        
        # Return the final solution and cost
        return final_answer, self.llm.cost_manager.total_cost

def extract_model_answer(text: str) -> str:
    pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
    boxed_matches = re.findall(pattern, text, re.DOTALL)
    if boxed_matches:
        return boxed_matches[-1].strip()

    # Fallback to last sentence extraction
    sentence_end_pattern = r"(?<!\\d)[.!?]\\s+"
    sentences = re.split(sentence_end_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences[-1] if sentences else ""
