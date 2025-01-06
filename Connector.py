import os
import warnings
import torchvision
import Prompt
import random

# Set environment variables for Hugging Face model directory and CUDA settings
os.environ["HF_HOME"] = r"D:\Stable Diffusion\HuggingFace\HuggingFaceCache"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Disable symlink warnings for huggingface_hub

def supress_warnings():
    # Disable beta transforms warning in torchvision
    torchvision.disable_beta_transforms_warning()

    # Suppress warnings from torchvision and huggingface_hub
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")


def generate_unique_filename(filename):
    """
    Generates a unique filename by appending a number if the file already exists.
    
    Parameters:
        filename (str): The initial filename to check for uniqueness.
    
    Returns:
        str: A unique filename with a numerical suffix if necessary, ensuring a .png extension.
    """
    base, ext = os.path.splitext(filename)  # Split filename into name and extension
    if not ext:
        ext = ".png"  # Add .png if there’s no extension
    elif ext.lower() != ".png":
        base = f"{base}{ext}"
        ext = ".png"  # Ensure the extension is .png

    filename = f"{base}{ext}"
    counter = 1
    
    # Loop to check if file exists and append a number if necessary
    while os.path.exists(os.path.join("Images", filename)):
        filename = f"{base} {counter}{ext}"
        counter += 1

    return filename

def main(object = None, num_inference_steps = 1, guidance_scale = 1, internet = True, fast = True):
    # Supresses all unnesccesary warnings
    supress_warnings()

    # Picks a prompt for me from predefined prompt
    if object is None:
        object = random.choice(Prompt.prompt_list)
        prompt = random.choice(object[1:])
    else:
        filename = object[0]
        prompt = object[1]

    # Now import the module and call generate_image
    from StableDiffusion35 import generate_image
    image = generate_image(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, internet = internet, fast = fast)

    # Save the generated image to the specified filename
    filename = generate_unique_filename(object[0])
    # Ensure the "Images" folder exists
    os.makedirs("Images", exist_ok=True)
    image.save(os.path.join("Images", filename))
    print(f"Image saved as {filename}")