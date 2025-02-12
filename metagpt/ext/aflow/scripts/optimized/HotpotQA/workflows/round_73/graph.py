from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_73.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow: 
    def __init__(self, 
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
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)

        # Check if the preliminary response is detailed enough 
        if len(preliminary_response['response']) < 50:  
            clarification_response = await self.custom(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT) 
            preliminary_response['response'] += " " + clarification_response['response']

        review_response = await self.answer_generate(input=preliminary_response['response'])

        multiple_answers = [await self.custom(input=problem + f" Review: {review_response['thought']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        consolidated_response = " ".join([ans['response'] for ans in multiple_answers])
        
        # Score responses based on keyword accuracy
        scored_responses = sorted(multiple_answers, key=lambda x: self.score_response(x['response']), reverse=True)
        
        ensemble_response = await self.sc_ensemble(solutions=[ans['response'] for ans in scored_responses])
        
        # Validation step to ensure the response is adequate 
        if len(ensemble_response['response']) < 50:  
            valid_response = await self.custom(input=ensemble_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
            final_response = valid_response['response']
        else: 
            final_response = ensemble_response['response']
        
        # Final review step
        final_review_response = await self.answer_generate(input=final_response)

        return final_review_response['answer'], self.llm.cost_manager.total_cost
        
    def score_response(self, response: str) -> int:
        keywords = ["correct", "accurate", "relevant", "detailed"]
        return sum(word in response for word in keywords)  # Basic scoring for keywords present
