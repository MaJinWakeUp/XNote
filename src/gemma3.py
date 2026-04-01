import os
from typing import List, Optional, Union
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


class Gemma3Model:
    """Wrapper class for Gemma3 model inference."""
    
    def __init__(
        self,
        model_path: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 512,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Gemma3 model.
        
        Args:
            model_path: Path to the pretrained model
            max_new_tokens: Maximum number of tokens to generate
            torch_dtype: Torch data type for model weights
            device_map: Device mapping strategy
            device: Device to run inference on
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading Gemma3 model from {model_path}...")
        
        # Load model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map
        ).eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Model loaded successfully on {device}")
    
    def _process_images(
        self,
        input_images: Optional[Union[str, List[str], Image.Image, List[Image.Image]]]
    ) -> List[Union[str, Image.Image]]:
        """
        Process and normalize input images to a consistent format.
        
        Args:
            input_images: Single or list of image paths, URLs, or PIL.Image objects
            
        Returns:
            List of processed images (paths/URLs or PIL.Image objects)
        """
        if input_images is None:
            return []
        
        # Convert single image to list
        if isinstance(input_images, (str, Image.Image)):
            input_images = [input_images]
        
        processed = []
        for img in input_images:
            if isinstance(img, str):
                # Check if it's a URL
                if img.startswith(('http://', 'https://')):
                    processed.append(img)
                # Check if it's a local path
                elif os.path.exists(img):
                    # Load as PIL Image for local files
                    pil_img = Image.open(img).convert('RGB')
                    processed.append(pil_img)
                else:
                    print(f"Warning: Image not found: {img}")
            elif isinstance(img, Image.Image):
                processed.append(img)
            else:
                print(f"Warning: Unsupported image type: {type(img)}")
        
        return processed
    
    def _build_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[List[Union[str, Image.Image]]] = None,
    ) -> List[dict]:
        """
        Build messages in the format required by Gemma3.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            input_images: List of image paths, URLs, or PIL.Image objects
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        # Build user message content
        user_content = []
        
        # Add images
        if input_images:
            processed_images = self._process_images(input_images)
            for img in processed_images:
                user_content.append({"type": "image", "image": img})
        
        # Add text prompt
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        
        # Add user message
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content
            })
        
        return messages
    
    def generate(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[Union[str, List[str], Image.Image, List[Image.Image]]] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt/question
            input_images: Path(s) or URL(s) to input image(s), or PIL.Image(s)
            max_new_tokens: Override default max_new_tokens
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text response
        """
        # Build messages
        messages = self._build_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_images=input_images,
        )
        
        # Prepare inputs
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device, dtype=self.torch_dtype)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Build generation kwargs
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens or self.max_new_tokens,
                "do_sample": do_sample,
            }
            
            if do_sample:
                if temperature is not None:
                    generation_kwargs["temperature"] = temperature
                if top_p is not None:
                    generation_kwargs["top_p"] = top_p
                if top_k is not None:
                    generation_kwargs["top_k"] = top_k
            
            # Generate response
            with torch.inference_mode():
                generation = self.model.generate(**generation_kwargs)
                generation = generation[0][input_len:]
            
            # Decode response
            decoded = self.processor.decode(generation, skip_special_tokens=True)
            
            return decoded.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
