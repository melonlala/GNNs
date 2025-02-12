from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_22.prompt as prompt_custom
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
        # Generate multiple responses using the AnswerGenerate operator
        responses = await self.answer_generate(input=problem)
        
        # Generate alternative answers using a different prompt
        alternative_responses = await self.custom(input=problem, instruction=prompt_custom.ALTERNATIVE_PROMPT)
        
        # Combine original and alternative responses for review
        all_responses = responses + alternative_responses['response']
        
        # Review the generated responses for quality and clarity
        review_response = await self.custom(input="Please evaluate the following answers for clarity and conciseness, and suggest improvements: " + " ".join(resp['answer'] for resp in all_responses), instruction=prompt_custom.REVIEW_PROMPT)
        
        # Use ScEnsemble to select the best response from reviewed answers
        ensemble_response = await self.sc_ensemble(solutions=[resp['answer'] for resp in review_response['response']])
        
        # Return the best response and the total cost
        return ensemble_response['response'], self.llm.cost_manager.total_cost
