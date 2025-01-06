from huggingface_hub import login
from diffusers import StableDiffusion3Pipeline
import torch

# Ensure user is logged in
def login_to_huggingface(file_path=r"D:\Stable Diffusion\Hugging Face"):
    token, username, website = None, None, None
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.startswith("Token:"):
                    token = line.split("Token:")[1].strip()
                elif line.startswith("User_Name:"):
                    username = line.split("User_Name:")[1].strip()
                elif line.startswith("Website:"):
                    website = line.split("Website:")[1].strip()
        
        if token:
            login(token)
            print(f"Logged in as {username} on {website}")
        else:
            print("Token not found. Please check the file.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"Login failed:")

# Function to generate an image using Stable Diffusion 3.5 with customizable parameters
def generate_image(prompt, num_inference_steps=30, guidance_scale=5, internet = True, fast = True): 

    # Define the model
    medium_35 = "stabilityai/stable-diffusion-3.5-medium"
    
    try:
        # Attempt to load the model with Local enviornment
        pipe = StableDiffusion3Pipeline.from_pretrained(medium_35, torch_dtype=torch.float16, local_files_only=True).to("cuda")
        
    except Exception as e: 
        print("Failed to load Model from Local enviornment") 

        if internet  == False:
            return
        
        # Attempt login
        login_to_huggingface()
        
        # Try loading the model from local cache only
        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(medium_35, torch_dtype=torch.float16).to("cuda")
        except Exception as e_local:
            print(f"Couldn't connect to the Hub:")
            return  # Exit if model cannot be loaded
        else:
            print("Loaded Model from Hub:")
    else:
        print("Loaded Model from Local enviornment")
    
    finally:
    
        if fast == True:
            # Enable memory-saving options
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()
            pipe.enable_sequential_cpu_offload()

        # Clear cache before generating the image
        torch.cuda.empty_cache()

        # Generate the image with specified prompt and parameters
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    return image