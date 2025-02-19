import torch
import re
import json
from sentence_transformers import (
    SentenceTransformer,
    util
)
from evaluation_base import BaseEvaluator


evaluator_path = "medalpaca/medalpaca-7b"


class MedicalResponseEvaluator(BaseEvaluator):
    """
    Evaluator for assessing medical AI responses against reference answers.
    
    Attributes:
        similarity_model (SentenceTransformer): Model for computing semantic similarity.
    """

    def __init__(self):
        """
        Initializes the medical response evaluator with a specialized prompt template.
        """
        super().__init__(evaluator_path)

        self.prompt_template = """
Context:
You are a radiologist evaluating AI responses to medical imaging questions. Focus strictly on factual accuracy against the reference answer. You will only output an answer in a JSON format.

Approach:
Compare the REFERENCE (gold standard) and RESPONSE (AI output):
1. Does the response contain the correct clinical finding? (Yes/No)
2. Does it add unnecessary or incorrect information? (Yes/No)
3. Confidence in accuracy (1-5)

Guidelines:
- Consider medical synonyms and equivalent phrasings
- Allow different terminology for same finding
- Be strict about safety-critical discrepancies
- Ignore stylistic/grammatical differences
- Reference is authoritative short answer

QUESTION: {question}
REFERENCE: {reference}
RESPONSE: {response}

Return ONLY JSON: {{
  "contains_correct_finding": bool,
  "has_incorrect_info": bool,
  "confidence": int
}}
Answer:"""

        #self.similarity_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    def evaluate(self, question: str, reference: str, response: str):
        """
        Evaluates the AI-generated response against the reference answer.

        Args:
            question (str): The medical question asked.
            reference (str): The gold standard reference answer.
            response (str): The AI-generated response to evaluate.

        Returns:
            Str: string containing evaluation results.
        """
        try:
            prompt = self.prompt_template.format(
                question=question,
                reference=reference,
                response=response
            )

            result = self.generate_response(prompt)

            assistant_start = "Answer:"
            response_start = result.find(assistant_start)
            if response_start != -1:
                result = result[response_start + len(assistant_start) :].strip()

            return result

        except Exception as e:
            return {"error": str(e)}

    def _clinical_consistency_check(self, reference: str, response: str):
        """
        Checks semantic similarity between the reference and response.

        Args:
            reference (str): The correct medical answer.
            response (str): The AI-generated answer.

        Returns:
            bool: True if similarity exceeds threshold, else False.
        """
        ref_embed = self.similarity_model.encode(reference.lower().strip())
        resp_embed = self.similarity_model.encode(response.lower().strip())

        similarity = util.pytorch_cos_sim(ref_embed, resp_embed).item()
        return similarity > 0.7  # Clinical threshold
