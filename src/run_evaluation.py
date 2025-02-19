import json
import os
import torch
import gc
from tqdm import tqdm
from evaluation import MedicalResponseEvaluator
from model_utils import load_model_and_processor, generate_predictions
from data_utils import prepare_dataloader, load_vqa_rad_dataset

BASE_MODEL_PATH = "llava-hf/llava-1.5-7b-hf"


def evaluate_model(dataset, model_path, save_path="evaluation_results.json", is_peft=False):
    """
    Conducts model evaluation by generating predictions and comparing them against reference answers.

    - Optimizes GPU settings for faster inference.
    - Ensures answers are generated only if not previously saved.
    - Uses the `MedicalResponseEvaluator` to assess response accuracy.
    - Saves evaluation results in a JSON file.

    Args:
        dataset (list): A dataset containing image-question-answer pairs.
        model_path (str): Path to the model to be evaluated.
        save_path (str, optional): Path to save evaluation results. Defaults to "evaluation_results.json".
        is_peft (bool, optional): Indicates if the model uses Parameter-Efficient Fine-Tuning (PEFT). Defaults to False.

    Returns:
        None: Results are saved to the specified file.
    """

    torch.backends.cuda.matmul.allow_tf32 = True  # Enable tensor cores
    torch.backends.cudnn.benchmark = True        # Optimize convolution algorithms
    torch.cuda.empty_cache()

    # Load Data
    dataloader = prepare_dataloader(dataset)
    
    # Set path for results generation 
    gen_path = "generation_results.json" if not is_peft else "fin_generation_results.json"
    
    # Check if Answers has already been generated to avoid regenerating each time
    if not os.path.isfile(gen_path):
        # Load Model and processor
        model, processor = load_model_and_processor(model_path, is_peft)
        
        # Set model to evaluation mode
        model.eval()

        # Pre-warm GPU
        _ = model.generate(**processor(text="Warmup", return_tensors="pt").to("cuda"), max_new_tokens=1)
        
        # Use model on data
        predictions, references = generate_predictions(model, processor, dataloader)
        
        # Free up memory 
        del model, processor
        torch.cuda.empty_cache()
        gc.collect()

        with open(gen_path, "w") as f:
            json.dump({'predictions': predictions, 'references': references}, f, indent=4)
    else:
        with open(gen_path, "r") as f:
            gen = json.load(f)
        predictions, references = gen['predictions'], gen['references']
    
    torch.cuda.empty_cache()
    del dataloader

    # Instantiate a mecical evaluatior
    evaluator = MedicalResponseEvaluator()
    
    # Start evaluating model responses against reference answers
    evaluations = []
    for i, example in enumerate(tqdm(dataset, total=len(dataset))):
        question = example["question"]
        reference_answer = example['answer']
        predicted_answer = predictions[i]
        
        assistant_start = "ASSISTANT:"
        response_start = predicted_answer.find(assistant_start)
        predicted_answer = predicted_answer[response_start + len(assistant_start) :].strip()

        result = evaluator.evaluate(
            question=question,
            reference=reference_answer,
            response=predicted_answer
        )

        evaluations.append(result)

    with open(save_path, "w") as f:
        json.dump(evaluations, f, indent=4)

    print(f"✅ Evaluation completed. Results saved in {save_path}")


if __name__ == "__main__":

    datasets = load_vqa_rad_dataset()

    # Evaluate pretrained model
    evaluate_model(datasets["test"], BASE_MODEL_PATH, "evaluation_pretrained.json")

    # Evaluate fine-tuned model
    evaluate_model(datasets["test"], "./llava_finetuned/checkpoint-178", "evaluation_finetuned.json", is_peft=True)

