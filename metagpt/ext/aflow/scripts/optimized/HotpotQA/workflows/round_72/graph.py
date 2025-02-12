from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_72.prompt as prompt_custom
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
        self.self_ask = operator.SelfAsk(self.llm)

    async def __call__(self, problem: str): 
        """
        Implementation of the workflow 
        """ 
        preliminary_response = await self.custom(input=problem, instruction=prompt_custom.REVIEW_PROMPT)

        # Self-ask step for clarity and depth
        self_ask_response = await self.self_ask(input=preliminary_response['response'])

        # Check if the self-ask response is detailed enough 
        if len(self_ask_response['response']) < 50:  
            clarification_response = await self.custom(input=problem, instruction=prompt_custom.CLARIFICATION_PROMPT) 
            self_ask_response['response'] += " " + clarification_response['response']

        review_response = await self.answer_generate(input=self_ask_response['response'])

        multiple_answers = [await self.custom(input=problem + f" Review: {review_response['thought']}", instruction=prompt_custom.FINAL_PROMPT) for _ in range(3)]
        
        # Consolidate multiple responses to enhance result quality
        consolidated_response = " ".join([ans['response'] for ans in multiple_answers])
        ensemble_response = await self.sc_ensemble(solutions=[consolidated_response])
        
        # Validation step 
        if len(ensemble_response['response']) < 50:  
            valid_response = await self.custom(input=ensemble_response['response'], instruction=prompt_custom.VALIDATION_PROMPT)
            final_response = valid_response['response']
        else: 
            final_response = ensemble_response['response']
        
        final_review_response = await self.answer_generate(input=final_response)

        return final_review_response['answer'], self.llm.cost_manager.total_cost
