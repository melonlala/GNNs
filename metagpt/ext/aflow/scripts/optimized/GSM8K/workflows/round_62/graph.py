from typing import Literal
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.GSM8K.workflows.round_62.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]

class Workflow: 
    def __init__(self, name: str, llm_config, dataset: DatasetType) -> None: 
        self.name = name 
        self.dataset = dataset 
        self.llm = create_llm_instance(llm_config) 
        self.llm.cost_manager = CostManager() 
        self.custom = operator.Custom(self.llm) 
        self.programmer = operator.Programmer(self.llm) 
        self.sc_ensemble = operator.ScEnsemble(self.llm) 
        self.self_ask = operator.SelfAsk(self.llm)  # Added SelfAsk operator

    async def __call__(self, problem: str): 
        """ 
        Implementation of the workflow 
        """ 
        # Generate initial and alternative solutions using the custom operator. 
        initial_solution = await self.custom(input=f"Here is the problem: {problem}. Can you help me solve it?", instruction="") 
        additional_solution = await self.custom(input=f"Here is the problem: {problem}. Can you provide another way to find the solution?", instruction="") 
        
        # Validation phase: Use self-consistency to validate and select the best solution. 
        ensemble_response = await self.sc_ensemble(solutions=[initial_solution['response'], additional_solution['response']], problem=problem) 
        
        # Review phase: Verify the correctness of all provided solutions before finalizing.
        review_response = await self.custom(input=f"Review these solutions: {initial_solution['response']} and {additional_solution['response']}. Which one is correct?", instruction="") 

        # Provide analysis with the programmer for better clarity of solutions and their validation. 
        analysis_response = await self.programmer(problem=problem, analysis=f"Initial Solution: {initial_solution['response']}, Additional Solution: {additional_solution['response']}, Selected Solution: {ensemble_response['response']}, Review: {review_response['response']}") 

        # Double-check the interpretation of the problem via self-ask
        self_ask_response = await self.self_ask(input=problem + f" Analyze the solutions: {analysis_response['output']}", instruction="Clarify the main question to validate the final answer.")

        return self_ask_response['response'], self.llm.cost_manager.total_cost
