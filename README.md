# LLaVA Fine-Tuning and Evaluation Framework

This repository provides an end-to-end pipeline for fine-tuning and evaluating the **LLaVA (Large Language and Vision Assistant)** model on **VQA-RAD**, a medical visual question-answering dataset. The framework includes dataset handling, model fine-tuning using LoRA, and evaluation with custom metrics.

---

## **Project Structure**

```
LLAVA/
│── logslurms/                   # Logs for training and evaluation runs
│── notebooks/
│   └── training_analysis.ipynb  # Jupyter notebook for analyzing training logs
│── src/
│   │── data_utils.py         # Handles dataset loading and preprocessing
│   │── model_utils.py        # Functions for loading, modifying, and configuring models
│   │── fine_tuning.py        # LoRA fine-tuning pipeline
│   │── evaluation_base.py    # Base classes for evaluation
│   │── evaluation.py         # Concrete evaluation implementation
│   │── callbacks.py          # Custom training callbacks for monitoring training loss
│   │── run_evaluation.py     # Script to run evaluation and generate reports
│── terminal_results/         # Evaluation Results
│   │── evaluation_finetuned.json  # Evaluation results for the fine-tuned model
│   │── evaluation_pretrained.json # Evaluation results for the pretrained model
└── README.md                 # Documentation
```

---

## **Installation & Setup**

### **1. Install Dependencies**
Ensure you have Python 3.8+ and install the required packages:
```bash
pip install -r requirements.txt
```

### **2. Download the LLaVA Model**
The default model used is `llava-hf/llava-1.5-7b-hf`. If necessary, modify `MODEL_NAME` in `fine_tuning.py` accordingly.

---

## **Usage**

### **1. Fine-Tuning the Model**
To fine-tune LLaVA on VQA-RAD using LoRA, run:
```bash
python src/fine_tuning.py
```
This script:
- Loads the VQA-RAD dataset (`data_utils.py`)
- Loads and modifies the LLaVA model (`model_utils.py`)
- Applies LoRA for efficient fine-tuning
- Trains the model using `Trainer` from Hugging Face
- Logs losses using `LossLoggerCallback` (`callbacks.py`)
- Saves the fine-tuned model

### **2. Evaluating the Model**
Run evaluation using:
```bash
python src/run_evaluation.py
```
This script:
- Loads the fine-tuned model
- Generates predictions on test data
- Computes evaluation metrics using `evaluation.py`
- Saves results to `evaluation_finetuned.json`

To compare with the pretrained model:
```bash
python src/run_evaluation.py --pretrained
```
Results are stored in `evaluation_pretrained.json`.

---

## **Key Components**

### **1. `data_utils.py`**
- Loads the VQA-RAD dataset and processes images & text for LLaVA.
- Implements `CustomDataCollator` for efficient batch processing.

### **2. `model_utils.py`**
- Loads the LLaVA model and tokenizer.
- Applies LoRA (`apply_lora`) to fine-tune efficiently.

### **3. `fine_tuning.py`**
- Defines `fine_tune_model()`, which handles the full fine-tuning process.
- Uses mixed precision training (`fp16`) and LoRA adapters.
- Logs training loss using `LossLoggerCallback` (`callbacks.py`).

### **4. `evaluation.py`**
- Implements `MedicalResponseEvaluator`, which computes:
  - Clinical relevance metrics

### **5. `evaluation_base.py`**
- Defines `BaseEvaluator`, an abstract class for evaluation.
- `MedicalResponseEvaluator` extends this to implement custom evaluation logic.

### **6. `callbacks.py`**
- Implements `LossLoggerCallback` to track training loss and evaluation loss at each step.

### **7. `run_evaluation.py`**
- Loads the model and dataset.
- Generates predictions using `generate_predictions()`.
- Computes evaluation metrics using `MedicalResponseEvaluator`.
- Saves results as a JSON file.

---

## Evaluation Results

The model was evaluated using an automated LLM-based scoring system. Below are the comparative results:

| Metric                  | Pre-Trained Model | Fine-Tuned Model |
|-------------------------|------------------|------------------|
| total_responses        | 451.00           | 451.00           |
| valid_responses        | 426.00           | 407.00           |
| correct_rate          | 0.51             | 0.47             |
| incorrect_rate        | 0.42             | 0.48             |
| neutral_rate         | 0.07             | 0.05             |
| contradiction_rate    | 0.00             | 0.00             |
| invalid_format_rate   | 0.06             | 0.10             |
| average_confidence   | 4.89             | 4.76             |
| confidence_completeness | 1.00          | 1.00             |

### Key Takeaways
- The **correct rate** decreased from **51% to 47%**, while the **incorrect rate** increased from **42% to 48%**.
- The **invalid format rate** increased from **6% to 10%**, indicating more formatting inconsistencies.
- The **average confidence** of responses slightly dropped from **4.89 to 4.76**.
- **Contradiction rate remained at 0**, meaning the model did not produce directly conflicting responses.
  
The decline in performance may be attributed to several factors:  
- **Catastrophic forgetting**.
- - **Misalignment between fine-tuning objectives and evaluation methodology**: The evaluation was conducted using an LLM-as-a-judge framework. While this approach is justified given the nuanced nature of the model's answers and the limitations of standard techniques (ROUGE, BLEU, etc.) in performing soft matching, the judge may still produce misleading or inaccurate evaluations.  


---

## **Customization**

### **Modifying Fine-Tuning Hyperparameters**
Edit `training_args` in `fine_tuning.py`:
```python
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=100,
    evaluation_strategy="epoch",
)
```

### **Changing the Evaluation Metrics**
Modify `evaluate()` in `evaluation.py` to add custom metrics.

---

## **Contributing**
If you’d like to contribute:
1. Fork the repo
2. Create a new branch (`feature-xyz`)
3. Commit changes and open a PR

---

## **License**

This fine-tuned model is based on **LLaVA-1.5**, which in turn utilizes **LLaMA 2**. As such, it is subject to the **LLaMA 2 Community License Agreement**, which governs the use, distribution, and modification of the model.  

Additionally, LLaVA-1.5 has been trained using **GPT-4-generated multimodal instruction-following data**, which may impose further restrictions, particularly on commercial usage. Users must review and comply with any applicable terms related to both LLaMA 2 and GPT-4 data usage.  

### Key Licensing Considerations:  
- The fine-tuned model inherits the **LLaMA 2 Community License** and must be used in accordance with its terms.  
- Any deployment or distribution of this model should acknowledge the **original authors of LLaVA-1.5** and adhere to its licensing conditions.  
- If you plan to use this model commercially, ensure compliance with **both Meta’s LLaMA 2 license and OpenAI's policies** regarding GPT-4-generated data.  

For full details, please refer to:  
- [LLaMA 2 Community License](https://github.com/facebookresearch/llama/blob/main/LICENSE)  
- [LLaVA Model Card and License](https://huggingface.co/liuhaotian/llava-v1.5-7b)  
- [Meta AI’s LLaMA 2 Announcement](https://ai.meta.com/llama/)  

By using this fine-tuned model, you agree to abide by the terms and conditions set forth in the above licenses.
