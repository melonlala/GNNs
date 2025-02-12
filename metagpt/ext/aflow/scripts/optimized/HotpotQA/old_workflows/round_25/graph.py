from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_25.prompt as prompt_custom
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
        # Self-ask to refine the problem statement
        clarification_response = await self.custom(input="What additional information is needed to clarify the question: " + problem, instruction=prompt_custom.CLARIFICATION_PROMPT)
        
        # Generate multiple responses using the AnswerGenerate operator
        responses = await self.answer_generate(input=clarification_response['response'])
        
        # Generate alternative answers for a broader perspective
        alternative_responses = await self.answer_generate(input="Provide alternative answers for: " + clarification_response['response'])
        
        # Review the generated responses for quality and clarity
        review_response = await self.custom(input="Please evaluate the following answers for clarity and conciseness, highlighting strengths and areas for improvement, and suggest the best answer among them: " + " ".join(resp['answer'] for resp in responses + alternative_responses), instruction=prompt_custom.REVIEW_PROMPT)
        
        # Generate a summary of the best-reviewed answers
        summary_response = await self.custom(input="Summarize the best-reviewed answers: " + " ".join(resp['answer'] for resp in review_response['response']), instruction=prompt_custom.SUMMARY_PROMPT)
        
        # Use ScEnsemble to select the best response from the summarized answers
        ensemble_response = await self.sc_ensemble(solutions=[resp['answer'] for resp in summary_response['response']])
        
        # Return the best response and the total cost
        return ensemble_response['response'], self.llm.cost_manager.total_cost
