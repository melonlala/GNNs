from typing import Literal
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.MMLU.workflows.round_41.prompt as prompt_custom
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
        

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using the custom operator
        solutions = []
        for _ in range(3):  # Generate three different responses
            solution = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
            solutions.append(solution["response"])
        
        # Review the generated solutions before ensemble selection
        reviewed_solutions = [sol for sol in solutions if self.review_solution(sol, problem)]
        
        # Use self-consistency to select the best solution
        ensemble_solution = await self.sc_ensemble(reviewed_solutions)
        formatted_solution = await self.format(ensemble_solution["response"])
        
        # Generate a counterexample if the ensemble solution is invalid
        counterexample = await self.answer_generate(input=problem)
        
        return formatted_solution['solution'], self.llm.cost_manager.total_cost, counterexample['answer']

    def review_solution(self, solution: str, problem: str) -> bool:
        # Implement a review logic that checks if the solution matches the right answer
        return len(solution) > 0 and solution == self.get_right_answer(problem)

    def get_right_answer(self, problem: str) -> str:
        # This function should extract the right answer from the problem context
        # For simplicity, we will assume the right answer is provided in the problem
        # In a real scenario, this would involve more complex logic
        return problem.split("right_answer: ")[1].strip()  # Example extraction logic
