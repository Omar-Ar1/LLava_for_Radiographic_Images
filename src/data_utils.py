import numpy as np
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader

class CustomDataCollator:
    """
    Handles proper batching of multimodal inputs by processing text and image data 
    together for model training. It formats prompts, tokenizes text, and ensures 
    correct padding and labeling.

    Attributes:
        processor: A tokenizer/processor that handles text and image inputs.
    """
    def __init__(self, processor):
        """
        Initializes the CustomDataCollator with a processor.

        Args:
            processor: A processor object (such as LlavaProcessor) that 
                      handles tokenization and image transformations.
        """
        self.processor = processor
        
    def __call__(self, examples):
        """
        Processes and collates a batch of multimodal examples.

        Args:
            examples (list of dict): A batch of examples where each example 
                                     contains an 'image', 'question', and 'answer'.

        Returns:
            dict: A dictionary containing:
                - 'input_ids' (torch.Tensor): Tokenized input text.
                - 'attention_mask' (torch.Tensor): Attention mask for padding.
                - 'pixel_values' (torch.Tensor): Preprocessed image tensors.
                - 'labels' (torch.Tensor): Target labels for training.
        """
        # Separate out text vs. image fields
        images = []
        texts = []
        for example in examples:
          image, question, answer =  example['image'], example['question'], example['answer']
          images.append(image)
          prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
          texts.append(prompt)

        # Collate text part
        text_batch = self.processor(text=texts, images=images,  padding='longest', max_length=384, return_tensors='pt')

        # Collate images
        labels = text_batch['input_ids'].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        text_batch['labels'] = labels

        # Return a dictionary instead of a tuple
        return {
            'input_ids': text_batch['input_ids'],
            'attention_mask': text_batch['attention_mask'],
            'pixel_values': text_batch['pixel_values'],
            'labels': text_batch['labels']
        }
    

def load_vqa_rad_dataset(split_ratio=0.8, seed=42):
    """
    Loads the 'flaviagiammarino/vqa-rad' dataset from Hugging Face and splits it into
    train, validation, and test sets.

    Args:
        split_ratios (tuple): Ratios for splitting the dataset (train, validation, test).
        seed (int): Random seed for reproducibility.

    Returns:
        DatasetDict: A dictionary-like Hugging Face DatasetDict containing 'train', 'valid', and 'test' splits.
    """
    
    # Load dataset from Hugging Face
    dataset = load_dataset("flaviagiammarino/vqa-rad")

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print(len(test_dataset))
    # Split training dataset into train and validation
    split = train_dataset.train_test_split(train_size=split_ratio, seed=seed, shuffle=True)
    train_dataset = split['train']
    val_dataset = split['test']

    # Return as a DatasetDict to allow .map() operations
    return {
        "train": train_dataset,
        "valid": val_dataset,
        "test": test_dataset
    }



# Data Loading
def prepare_dataloader(dataset, batch_size=16):
    """
    Creates a DataLoader for efficient loading of multimodal data, utilizing pinned memory, 
    parallel workers, and prefetching to optimize performance.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch. Default is 16.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance with optimized settings.
    """
   
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,  # Faster data transfer to GPU
        num_workers=4,    # Parallel data loading
        persistent_workers=True, # Ensure workers are alive after each epoch
        prefetch_factor=4, # Helps keep the data loading pipeline filled (GPU doesn't wait for data)
        collate_fn=lambda batch: {
            'image': [item['image'] for item in batch],
            'question': [item['question'] for item in batch],
            'answer': [item['answer'] for item in batch]
        }
    )
