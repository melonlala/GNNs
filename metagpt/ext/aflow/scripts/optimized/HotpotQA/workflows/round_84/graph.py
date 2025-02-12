from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_84.prompt as prompt_custom
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
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple preliminary answers for diversity
        preliminary_responses = [await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT) for _ in range(3)]
        
        # Consolidate preliminary responses to enhance variety
        consolidated_preliminary_response = " ".join([resp['response'] for resp in preliminary_responses])
        
        # Review the consolidated preliminary response
        review_response = await self.answer_generate(input=consolidated_preliminary_response)
        
        # Generate multiple answers for self-consistency
        multiple_answers = [await self.custom(input=problem + f" Review: {review_response['thought']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        # Consolidate responses to enhancing self-consistency input
        consolidated_response = " ".join([ans['response'] for ans in multiple_answers])
        
        # Use self-consistency to select the best solution based on consolidated responses
        ensemble_response = await self.sc_ensemble(solutions=[consolidated_response])
        
        # Cross-check the ensemble response for accuracy with a secondary review
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        
        return final_review_response['answer'], self.llm.cost_manager.total_cost
