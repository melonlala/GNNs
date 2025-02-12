from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_123.prompt as prompt_custom
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

        if len(preliminary_response['response']) < 50:  
            clarification_response = await self.custom(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT) 
            preliminary_response['response'] += " " + clarification_response['response']

        review_response = await self.answer_generate(input=preliminary_response['response'])

        multiple_answers = [await self.custom(input=problem + f" Review: {review_response['thought']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        peer_review_scores = [self.score_response(ans['response']) for ans in multiple_answers]
        best_answer_index = peer_review_scores.index(max(peer_review_scores))
        
        consolidated_response = multiple_answers[best_answer_index]['response']
        
        ensemble_response = await self.sc_ensemble(solutions=[consolidated_response])
        
        if len(ensemble_response['response']) < 50:  
            valid_response = await self.custom(input=ensemble_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
            final_response = valid_response['response']
        else: 
            final_response = ensemble_response['response']
        
        final_review_response = await self.answer_generate(input=final_response)

        return final_review_response['answer'], self.llm.cost_manager.total_cost

    def score_response(self, response: str) -> int:
        # Simulated scoring function based on factual accuracy, relevance, and completeness.
        score = 0
        if "accurate" in response:
            score += 1
        if "relevant" in response:
            score += 1
        if "complete" in response:
            score += 1
        return score
