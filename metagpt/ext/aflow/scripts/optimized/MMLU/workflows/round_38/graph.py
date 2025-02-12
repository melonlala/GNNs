from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.round_38.prompt as prompt_custom
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
        self.sc_ensemble = operator.ScEnsemble(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)
        self.format = operator.Format(self.llm)
        

    async def __call__(self, problem: str, right_answer: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using the custom operator
        solutions = []
        for _ in range(3):  # Generate three different responses
            solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
            solutions.append(solution["response"])
        
        # Review the generated solutions before ensemble selection
        reviewed_solutions = [sol for sol in solutions if self.review_solution(sol, right_answer)]
        
        # Use self-consistency to select the best solution
        ensemble_solution = await self.sc_ensemble(reviewed_solutions)
        
        # Generate a counterexample if the ensemble solution is incorrect
        if ensemble_solution["response"] != right_answer:
            counterexample = await self.answer_generate(input=problem)
            print(f"Counterexample generated: {counterexample['answer']}")
        
        formatted_solution = await self.format(ensemble_solution["response"])
        return formatted_solution['solution'], self.llm.cost_manager.total_cost

    def review_solution(self, solution: str, right_answer: str) -> bool:
        # Implement a review logic to check correctness against the right answer
        return solution == right_answer  # Only accept solutions that match the right answer
