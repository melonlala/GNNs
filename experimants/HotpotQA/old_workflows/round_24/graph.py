from typing import Literal
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.HotpotQA.workflows.round_24.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

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
        # Generate multiple detailed answers using the AnswerGenerate operator
        alternative_answers = await self.answer_generate(input=problem)
        
        # Use the custom method to generate the initial response
        initial_solution = await self.custom(input=problem + f" Details: {alternative_answers['thought']}", instruction="")
        
        # Generate a summary of the initial solution for context
        summary_solution = await self.custom(input=initial_solution['response'], instruction="Summarize this answer.")
        
        # Generate follow-up questions based on the initial solution
        follow_up_questions = await self.custom(input=initial_solution['response'], instruction="Generate follow-up questions based on this answer.")
        
        # Review the initial solution for accuracy, including the summary, alternative answers, and follow-up questions
        review_solution = await self.custom(input=initial_solution['response'], instruction=f"Review this answer for accuracy. Summary: {summary_solution['response']}. Alternative Answers: {alternative_answers['answer']}. Follow-up Questions: {follow_up_questions['response']}")
        
        # Generate a final review of the reviewed solution
        final_review = await self.custom(input=review_solution['response'], instruction="Provide a final review of this answer.")
        
        # Use ScEnsemble to select the best solution from multiple responses
        ensemble_solution = await self.sc_ensemble(solutions=[initial_solution['response'], final_review['response']])
        
        return ensemble_solution['response'], self.llm.cost_manager.total_cost
