from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_82.prompt as prompt_custom
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

        # Check if the preliminary response is unclear
        if "unclear" in preliminary_response['response']:
            clarification_response = await self.custom(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
            preliminary_response = await self.custom(input=clarification_response['response'], instruction=prompt_custom.REVIEW_PROMPT)
        
        # Review the preliminary response
        review_response = await self.answer_generate(input=preliminary_response['response'])
        
        # Generate a summary of the review
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)

        # Review the summary to ensure its clarity before generating multiple answers
        review_summary_response = await self.answer_generate(input=summary_response['response'])

        # Generate multiple answers considering the review and summary
        multiple_answers = [await self.custom(input=problem + f" Review: {review_response['thought']} Summary: {review_summary_response['answer']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]

        # Review each generated answer for accuracy and relevance
        reviewed_answers = [await self.answer_generate(input=ans['response']) for ans in multiple_answers if "relevant" in ans['response']]
        
        # Use self-consistency to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[ans['answer'] for ans in reviewed_answers])
        
        # Review the ensemble response for accuracy
        final_review_response = await self.answer_generate(input=ensemble_response['response'])
        
        return final_review_response['answer'], self.llm.cost_manager.total_cost
