from model_utils import load_evaluator
import torch
import gc
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class BaseEvaluator:
    """
    Base class for evaluation, handling model initialization and response generation.

    Attributes:
        model_name (str): Name of the pre-trained model to use for evaluation.
        tokenizer (AutoTokenizer): Tokenizer for processing text inputs.
        model (AutoModelForCausalLM): Loaded model for generating responses.
        prompt_template (str): Template for formatting the evaluation prompt.
    """

    def __init__(self, model_name: str):
        """
        Initializes the evaluator by loading the tokenizer and model.

        Args:
            model_name (str): Name of the pre-trained model to load.
        """
        self.model, self.tokenizer = load_evaluator(model_name=model_name)
        self.prompt_template = None  # To be defined in subclasses

    def generate_response(self, prompt: str, max_new_tokens: int = 250):
        """
        Generates a response based on the given prompt using the loaded model.

        Args:
            prompt (str): The input prompt for the model.
            max_new_tokens (int): Maximum number of tokens to generate.

        Returns:
            str: The model-generated response.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.model.device)

        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
