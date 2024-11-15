import os
import glob
import torch

def save_model(model, path, global_step, start_iteration):

    torch.save({
        'model_state_dict': model.state_dict(),
        'global_step': global_step,
        'start_iteration': start_iteration
    }, path)


def load_model(model, path):
    # Check if the checkpoint file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    # Load the checkpoint
    checkpoint = torch.load(path)

    # Check if 'model_state_dict' exists in the checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise KeyError(f"'model_state_dict' not found in the checkpoint at {path}")

    model.eval()  # Set the model to evaluation mode
    return checkpoint  # Return the checkpoint to access global_step

def get_latest_checkpoint(dir_path):
    # Get a list of all checkpoint files in the directory
    checkpoint_files = glob.glob(os.path.join(dir_path, '*.pth'))
    if not checkpoint_files:
        return None
    # Sort the files by modification time and get the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    return latest_checkpoint

