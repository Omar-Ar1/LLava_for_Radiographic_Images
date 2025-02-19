import torch
import re
import json
import os
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import infer_auto_device_map
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoTokenizer
)

from peft import(
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration

BASE_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"

# Define Custom Class
class CustomLlavaForConditionalGeneration(LlavaForConditionalGeneration):
    """
    A custom subclass of LlavaForConditionalGeneration that overrides the forward 
    method to dynamically handle `num_logits_to_keep`, ensuring flexible generation 
    behavior.

    This class ensures that `num_items_in_batch` (if passed) is ignored and that 
    `num_logits_to_keep` is dynamically determined based on `input_ids` when not explicitly set.

    Args:
        LlavaForConditionalGeneration: The base class for conditional generation.
    """
    def forward(self, *args, num_items_in_batch=None, num_logits_to_keep=None, **kwargs):
        """
        Overrides the forward method to manage `num_logits_to_keep` dynamically.

        Args:
            *args: Positional arguments for the base class forward method.
            num_items_in_batch (int, optional): Ignored if provided.
            num_logits_to_keep (int, optional): Specifies how many logits to keep.
                                                If None, it is set to the sequence length of `input_ids`.
            **kwargs: Additional keyword arguments for the base class forward method.

        Returns:
            torch.Tensor: The output logits from the model.
        """
        # Ignore `num_items_in_batch` if passed
        kwargs.pop('num_items_in_batch', None)

        # Dynamically set num_logits_to_keep
        if num_logits_to_keep is None and "input_ids" in kwargs:
            num_logits_to_keep = kwargs["input_ids"].shape[1] if kwargs["input_ids"].shape[1] > 0 else 1
            
        return super().forward(*args, num_logits_to_keep=num_logits_to_keep, **kwargs)



def load_model_and_processor(model_path, is_peft=False):
    """
    Loads a LLaVA model and processor, handling both base and fine-tuned (PEFT) models with quantization.

    This function ensures proper model loading with 4-bit quantization for efficient inference, 
    while also adapting the tokenizer for medical imaging applications.

    Args:
        model_path (str): Path to the model checkpoint (either base or fine-tuned).
        is_peft (bool, optional): If True, loads a PEFT fine-tuned model and merges adapters.
                                  If False, loads a standard pretrained model. Default is False.

    Steps:
        1. Configures 4-bit quantization for memory efficiency.
        2. Loads either:
            - A full pretrained or fine-tuned model (if `is_peft=False`).
            - A base model with PEFT adapters, which are merged for inference (if `is_peft=True`).
        3. Loads the processor (tokenizer + feature extractor).
        4. Ensures the tokenizer contains necessary special tokens for medical QA.
        5. Resizes token embeddings to account for added special tokens.

    Returns:
        Tuple[torch.nn.Module, transformers.Processor]: 
            - The loaded LLaVA model.
            - The corresponding processor (tokenizer + feature extractor).
    """

    # Configure quantization for medical imaging use cases
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    if is_peft:
        # Load base model + PEFT adapter
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for inference


    else:
        # Load full model (pretrained or merged fine-tuned)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    # Load processor with medical imaging adaptations
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=True,
        trust_remote_code=True
    )

    # Handle special tokens for medical QA
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if "<image>" not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
    model.resize_token_embeddings(len(tokenizer))

    return model, processor


# Optimized Prediction Generation
def generate_predictions(model, processor, dataloader):
    """
    Generates predictions for a batch of radiology images using an optimized inference pipeline.

    This function ensures efficient memory usage and stable performance while processing 
    medical image-question pairs in batches.

    Args:
        model (torch.nn.Module): The LLaVA model for vision-language generation.
        processor (transformers.Processor): The processor for handling images and tokenization.
        dataloader (torch.utils.data.DataLoader): DataLoader containing image-question-answer pairs.

    Steps:
        1. Moves the model to CUDA for efficient GPU computation.
        2. Iterates over batches in the dataloader.
        3. Constructs structured prompts following a radiology-specific approach.
        4. Processes images and text using the processor, ensuring correct tensor formatting.
        5. Moves inputs to CUDA for faster inference.
        6. Generates predictions with optimized parameters:
            - `max_new_tokens=300`: Limits output length.
            - `num_beams=1, do_sample=False`: Uses greedy decoding.
            - `use_cache=True, temperature=0.0`: Ensures deterministic output.
        7. Moves generated token IDs back to CPU and decodes them into text.
        8. Periodically clears CUDA cache and collects garbage to optimize memory usage.

    Returns:
        Tuple[List[str], List[str]]: 
            - A list of model-generated responses.
            - A list of ground truth answers for evaluation.
    """

    predictions, references = [], []
    
    model.to("cuda")  # Ensure model is on CUDA
    
    with torch.inference_mode(), torch.amp.autocast("cuda"):
        for batch_idx, batch in enumerate(dataloader):
            prompts = [
    f"""USER: <image>
You are a radiology doctor. Your task is to anlayze radiographic images, and to answer the given question in a coherent sentence. You cannot answer with just yes or no, your answer needs to be a well formulated sentence.
Question: {q}

Approach:
1. Carefully observe the entire image and identify anatomical structures
2. Note any abnormal findings (e.g., lesions, fractures, opacities)
3. Consider differential diagnoses for observed abnormalities
4. Correlate findings with clinical question
5. Formulate response using medically precise terms

Example:

Analysis
- Identification of right lower lobe consolidation
- Air bronchograms present
- No pleural effusion observed
- Consistent with bacterial pneumonia

Conclusion:
ASSISTANT: The chest radiograph demonstrates consolidation in the right lower lobe with air bronchograms, consistent with bacterial pneumonia. No pleural effusion is seen.
    
**Current Case:**
Question: {q}
ASSISTANT:""" 
    for q in batch['question']
]

            
            inputs = processor(
                images=batch['image'],
                text=prompts,
                padding=True,
                return_tensors='pt'
            )

            # Move all inputs to CUDA
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

            # Optimized generation parameters
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                temperature=0.0
            )

            # Move results back to CPU before decoding
            batch_preds = processor.batch_decode(
                generated_ids.cpu(), 
                skip_special_tokens=True
            )
            
            predictions.extend(batch_preds)
            references.extend(batch['answer'])

            # Free up memory
            del inputs, generated_ids
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() # Clears cache
                gc.collect() # Freeing Python-level memory

    return predictions, references


def load_evaluator(model_name="medalpaca/medalpaca-7b"):
    """Load a pre-trained model and tokenizer with medical-specific settings."""
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        legacy=False,
        padding_side="left",
        truncation_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer



# LoRA Configuration for Parameter-Efficient Fine-Tuning
def apply_lora(model):
    """
    Applies LoRA (Low-Rank Adaptation) fine-tuning to the model for efficient training.

    This function prepares the model for low-bit training and injects LoRA adapters into 
    specific layers, enabling parameter-efficient fine-tuning without modifying the full 
    model weights.

    Returns:
        model (torch.nn.Module): The model with LoRA adapters applied.

    Notes:
        - `prepare_model_for_kbit_training(model)`: Ensures compatibility with 4-bit quantization.
        - `LoraConfig`: Defines LoRA parameters such as rank, dropout, and target layers.
        - `get_peft_model`: Wraps the model with LoRA layers for selective fine-tuning.
        - LoRA is applied to all linear layers (`target_modules='all-linear'`).
    """

    # 1. Prepare LLaVA for k-bit (4-bit) fine-tuning
    model = prepare_model_for_kbit_training(model)

    # 2. Define LoRA Config
    lora_config = LoraConfig(
        r=16,  # Rank for low-rank adaptation
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear',  # LoRA applied to attention layers
        init_lora_weights='gaussian'
    )
    
    #3. Inject LoRA modules
    model = get_peft_model(model, lora_config)

    return model