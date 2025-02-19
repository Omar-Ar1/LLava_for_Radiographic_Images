import torch
from transformers import (
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainerCallback
)
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
from peft import(
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from datasets import DatasetDict
from data_utils import load_vqa_rad_dataset, CustomDataCollator
import math
from model_utils import load_model_and_processor, CustomLlavaForConditionalGeneration, apply_lora
from callbacks import LossLoggerCallback

# Model & Tokenizer Setup
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"  


# Training Function
def fine_tune_model():
    """
    Fine-tunes the LLaVA model on the VQA-RAD dataset using LoRA for parameter-efficient training.

    This function handles the full fine-tuning pipeline, including dataset loading, model 
    initialization, LoRA application, and setting up the Hugging Face `Trainer` with appropriate 
    training arguments. It also logs the loss throughout training and saves the fine-tuned model.

    Steps:
        1. Loads the VQA-RAD dataset.
        2. Initializes the LLaVA model and tokenizer with quantization.
        3. Applies LoRA fine-tuning.
        4. Configures training arguments, including batch size, learning rate, and evaluation steps.
        5. Sets up the Hugging Face `Trainer` with a custom data collator and loss logger.
        6. Runs the training loop.
        7. Saves the fine-tuned model and processor.

    Notes:
        - Uses LoRA for efficient fine-tuning while keeping the base model in 4-bit quantization.
        - Training uses mixed precision (`fp16`) for memory efficiency.
        - Saves the best model based on evaluation loss (`metric_for_best_model="loss"`).
        - Logs training loss every 100 steps.
        - The model is saved at `./llava_finetuned_2` after training.

    Returns:
        None
    """

    # Load dataset
    datasets = load_vqa_rad_dataset()
    
    # Load model and Tokenizer
    model, processor = load_model_and_processor(MODEL_NAME)
    model = apply_lora(model)
    
    # Instantiate a loss logger
    loss_logger = LossLoggerCallback()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./llava_finetuned",
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,   
        gradient_accumulation_steps=4,  # Effective batch size = 16 (GPU-dependent)
        remove_unused_columns=False,  # Required for custom datasets
        lr_scheduler_type='cosine',
        eval_strategy="steps",  
        save_strategy="steps",
        save_steps=89,  
        eval_steps=89, 
        save_total_limit=1,  # Keep last best checkpoint
        load_best_model_at_end=True, 
        metric_for_best_model="loss",
        greater_is_better=False,  # Lower loss is better
        logging_dir="./logs",
        logging_steps=100,  # Keep logging every 100 steps
        learning_rate=2e-4,
        warmup_steps=89,
        weight_decay=0.01,
        optim="adamw_torch",
        fp16=True,  # mixed precision for memory efficiency
        max_steps=712,  
        report_to="none",
        dataloader_num_workers=4, 
    )

    # Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["valid"],
        data_collator=CustomDataCollator(processor),
        callbacks=[loss_logger],
    )
    
    model.config.use_cache = False  # needed for gradient checkpointing sometimes
    
    # Start training
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("./llava_finetuned")
    processor.save_pretrained("./llava_finetuned")

    print("Fine-tuning completed! Model saved.")

if __name__ == "__main__":
    fine_tune_model()

