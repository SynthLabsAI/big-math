import asyncio
import os
import signal

import openai
from tqdm.asyncio import tqdm_asyncio


class OpenAIServerManager:
    def __init__(
        self, model_name, start_port=2341, tp=1, max_time=600, trust_remote_code=False,
    ):
        self.model_name = model_name
        self.start_port = start_port
        self.tp = tp
        self.max_time = max_time
        self.trust_remote_code = trust_remote_code

    def __enter__(self):
        self.ports, self.subprocs = self.launch_servers(
            self.model_name, self.start_port, self.tp, self.max_time,
        )
        self.async_clients = [
            openai.AsyncClient(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
            for port in self.ports
        ]
        self.sems = [asyncio.Semaphore(64) for _ in self.ports]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for proc in self.subprocs:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    def launch_servers(
        self, model_name, start_port=8001, tp=1, max_time=600
    ):
        raise NotImplementedError

    async def get_chat_responses(self, chats, **kwargs) -> list[openai.ChatCompletion]:
        """
        Get responses from the sglang server with retry functionality
        :param chats: list of chats
        :return: list of chat completions
        """
        responses = list()


        async def response_wrapper(client: openai.AsyncClient, sem, **kwargs):
            async with sem:
                try:
                    out = await client.chat.completions.create(**kwargs)
                    completions = [choice.message.content for choice in out.choices]
                    messages = kwargs["messages"]
                    messages[-1]["responses"] = completions
                    return messages
                except Exception as e:
                    print(f"Error in response_wrapper: {str(e)}")
                    default_message = ""
                    messages = kwargs["messages"]
                    messages[-1]["responses"] = [default_message]
                    return messages

        for i, chat in enumerate(chats):
            curr_kwargs = kwargs.copy()
            curr_kwargs["model"] = "default"
            curr_kwargs["messages"] = chat
            if "max_tokens" not in curr_kwargs:
                curr_kwargs["max_tokens"] = (
                    2048  # They can go realllyy reallly long if you just yolo it.
                )
            responses.append(
                response_wrapper(
                    self.async_clients[i % len(self.ports)],
                    self.sems[i % len(self.ports)],
                    **curr_kwargs,
                )
            )
        return await tqdm_asyncio.gather(*responses)

    async def get_completion_responses(
        self, prompts, **kwargs
    ) -> list[openai.Completion]:
        """
        Get responses from the sglang server
        :param prompts: list of prompts
        :return:
        """
        responses = list()

        async def response_wrapper(client: openai.AsyncClient, sem, **kwargs):
            async with sem:
                completions = await client.completions.create(**kwargs)
                completions = [choice.text for choice in completions.choices]
                kwargs["responses"] = completions
                return kwargs

        for i, prompt in enumerate(prompts):
            curr_kwargs = kwargs.copy()
            curr_kwargs["model"] = "default"
            curr_kwargs["prompt"] = prompt
            if "max_tokens" not in curr_kwargs:
                curr_kwargs["max_tokens"] = (
                    2048  # They can go realllyy reallly long if you just yolo it.
                )
            responses.append(
                response_wrapper(
                    self.async_clients[i % len(self.ports)],
                    self.sems[i % len(self.ports)],
                    **kwargs,
                )
            )
        return await tqdm_asyncio.gather(*responses)
