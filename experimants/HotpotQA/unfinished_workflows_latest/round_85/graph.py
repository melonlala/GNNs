from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_85.prompt as prompt_custom
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
        # Generate a detailed context for the problem
        context_response = await self.custom(input=problem, instruction=prompt_custom.CONTEXT_PROMPT)

        # Adding a request for a specific example
        example_response = await self.custom(input=problem, instruction=prompt_custom.EXAMPLE_PROMPT)

        # Generate a preliminary answer
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)

        # Check if the final response needs clarification
        if len(context_response['response']) < 50:  # Check if context is brief
            clarification_response = await self.custom(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
            context_response['response'] += " " + clarification_response['response']
        
        # Review the preliminary response with a more detailed analysis
        review_response = await self.answer_generate(input=preliminary_response['response'])
        
        # Generate a summary of the review
        summary_response = await self.custom(input=review_response['thought'], instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Generate multiple answers for self-consistency
        multiple_answers = [
            await self.custom(input=problem + f" Context: {context_response['response']} Example: {example_response['response']} Review: {review_response['thought']} Summary: {summary_response['response']}", instruction=prompt_custom.FINAL_PROMPT) 
            for _ in range(3)
        ]
        
        # Use self-consistency to select the best solution
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in multiple_answers])
        
        # Final verification of the ensemble response for accuracy
        final_verification_response = await self.answer_generate(input=ensemble_response['response'])
        
        # Check if the final answer has sufficient detail
        if len(final_verification_response['answer']) > 30:  # For example, ensure it is descriptive
            return final_verification_response['answer'], self.llm.cost_manager.total_cost
        else:
            return "Inconclusive", self.llm.cost_manager.total_cost
