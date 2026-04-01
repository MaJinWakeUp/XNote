import os
from typing import List, Optional, Union
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class Qwen3VLModel:
    """Wrapper class for Qwen3-VL model inference."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_new_tokens: int = 128,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        use_flash_attention: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the Qwen3-VL model.
        
        Args:
            model_path: Path to the pretrained model
            max_new_tokens: Maximum number of tokens to generate
            device_map: Device mapping strategy
            torch_dtype: Torch data type for model weights
            use_flash_attention: Whether to use flash attention
            device: Device to run inference on
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.use_flash_attention = use_flash_attention
        
        print(f"Loading Qwen3-VL model from {model_path}...")
        
        # Load model with optional flash attention
        if use_flash_attention:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device_map,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
            )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Model loaded successfully on {device}")
    
    def generate(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[Union[str, List[str], Image.Image, List[Image.Image]]] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            system_prompt: System prompt (incorporated into messages)
            user_prompt: User prompt/question
            input_images: Path(s) or URL(s) to input image(s), or PIL.Image(s)
            max_new_tokens: Override default max_new_tokens
            
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
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                **kwargs,
            )
            
            # Trim and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return ""
    
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
                # Check if it's a URL or local path
                if img.startswith(('http://', 'https://')):
                    processed.append(img)
                elif os.path.exists(img):
                    processed.append(img)
                else:
                    print(f"Warning: Image not found: {img}")
            elif isinstance(img, Image.Image):
                # PIL.Image object
                processed.append(img)
            else:
                print(f"Warning: Unsupported image type: {type(img)}")
        
        return processed

    def _build_messages(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[Union[str, List[str]]] = None,
    ) -> List[dict]:
        """
        Build messages in the format required by Qwen3-VL.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            input_images: Image path(s) or URL(s)
            
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
                user_content.append({
                    "type": "image",
                    "image": img
                })
        
        # Add text prompt
        if user_prompt:
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
        
        # Add user message
        if user_content:
            messages.append({
                "role": "user",
                "content": user_content
            })
        
        return messages
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
