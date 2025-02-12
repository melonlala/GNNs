import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
from GDesigner.llm.format import Message
from GDesigner.llm.price import cost_count
from GDesigner.llm.llm import LLM
from GDesigner.llm.llm_registry import LLMRegistry
import random
from openai import OpenAI, AsyncOpenAI
from dataclasses import asdict
import async_timeout
LM_STUDIO_URL = "http://localhost:1234/v1"
OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv(find_dotenv(), override=True)
MINE_BASE_URL = os.getenv('BASE_URL')
MINE_API_KEYS = os.getenv('API_KEY')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def gpt_achat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:

    api_kwargs: Dict[str, Any]
    if model == "lmstudio":
        api_kwargs = dict(base_url=LM_STUDIO_URL)
    else:
        api_key = MINE_API_KEYS
        base_url = MINE_BASE_URL
        api_kwargs = dict(api_key=api_key,base_url= base_url)

    aclient = AsyncOpenAI(**api_kwargs)
    # import pdb;pdb.set_trace()
    if not isinstance(messages[1],dict):
        formated_messages = [asdict(message) for message in messages]
    else: 
        formated_messages = messages
    try:
        async with async_timeout.timeout(1000):
            response = await aclient.chat.completions.create(model=model,
            messages=formated_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps)
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("GPT Timeout")
    if num_comps == 1:
        # cost_count(response, model)
        return response.choices[0].message.content
    # import pdb;pdb.set_trace()
    # cost_count(response, model)

    return [choice.message.content for choice in response.choices]

@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return await gpt_achat(self.model_name, messages)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
