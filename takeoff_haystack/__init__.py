from typing import Any, Dict, List, Optional

from haystack import component
from takeoff_client import TakeoffClient


@component
class TakeoffGenerator:
    """
    Generator based on Takeoff. Takeoff is a library from TitanML for running LLMs locally.
    This component provides an interface to generate text using a LLM running in Takeoff.
    """

    def __init__(
        self,
        base_url: str = "http://localhost",
        port: int = 3000,
        consumer_group: str = "primary",
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a TakeoffGenerator object.
        :param base_url: The base url on which takeoff is hosted.
        :param port: The port on which the takeoff instance is hosted.
        :param consumer_group: The consumer group that you want your request to be sent to. See https://docs.titanml.co/docs/Docs/model_management/readers
            for more information
        :param generation_kwargs: The generation kwargs to be used for the generation.
            These are the same as the ones used in the takeoff client: see https://docs.titanml.co/docs/apis/takeoff_client/#generate for more information
        """
        self.consumer_group = consumer_group
        self.generation_kwargs = generation_kwargs or {}
        self.client = TakeoffClient(base_url, port)

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run a Takeoff Model on the given prompt.
        :param prompt: The prompt to generate a response for.
        :param generation_kwargs: The generation kwargs supplied to the model. If not supplied here, the defaults from the `__init__` method are used.
        :return: The response from the model in haystack format
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        generation_kwargs.update({"consumer_group": self.consumer_group})

        response = self.client.generate(prompt, **generation_kwargs)

        response = {"replies": [response["text"]]}

        return response
