from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_105.prompt as prompt_custom
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
        # Generate a preliminary answer
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)
        
        # Review the preliminary response
        review_response = await self.answer_generate(input=preliminary_response['response'])
        
        # Generate a summary of the review
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Self-review of the primary answer before proceeding to generate multiple answers
        self_review_response = await self.answer_generate(input=preliminary_response['response'])

        # Generate multiple answers for self-consistency
        multiple_answers = [
            await self.custom(
                input=problem + f" Review: {review_response['thought']} Summary: {summary_response['response']} Self-Review: {self_review_response['thought']}",
                instruction=prompt_custom.FINAL_PROMPT
            ) for _ in range(3)
        ]
        
        # Review each generated answer for accuracy
        reviewed_answers = [await self.answer_generate(input=ans['response']) for ans in multiple_answers]
        
        # Review the answers again to ensure they are accurate and relevant
        final_reviewed_answers = [await self.answer_generate(input=ans['answer']) for ans in reviewed_answers]
        
        # Filter out irrelevant answers before self-consistency
        relevant_answers = [ans['answer'] for ans in final_reviewed_answers if 'relevant' in ans['thought'].lower()]
        
        # Use self-consistency to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=relevant_answers)
        
        # Review the ensemble response for accuracy
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        
        return final_review_response['answer'], self.llm.cost_manager.total_cost
