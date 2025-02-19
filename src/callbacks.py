from transformers import TrainerCallback

class LossLoggerCallback(TrainerCallback):
    """
    A custom callback for logging and tracking training and evaluation losses during training.

    This callback stores the loss values reported by the Hugging Face Trainer's logging mechanism.
    It appends losses from each logging step to `train_losses` and `eval_losses` lists.

    Attributes:
        train_losses (list): Stores training loss values at each logging step.
        eval_losses (list): Stores evaluation loss values at each logging step.

    Methods:
        on_log(args, state, control, logs=None, **kwargs):
            Logs training and evaluation losses whenever the Trainer logs metrics.
    """
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Collects and stores training and evaluation loss values when logging occurs.

        Args:
            args: Training arguments.
            state: The current state of the Trainer.
            control: Trainer control object.
            logs (dict, optional): Dictionary containing logged metrics.
            **kwargs: Additional arguments.

        Updates:
            - Appends the current training loss to `train_losses` if available.
            - Appends the current evaluation loss to `eval_losses` if available.
        """
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
