import os
from typing import List, Optional, Union
import torch
import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
from PIL import Image


class VILAModel:
    """Wrapper class for VILA model inference."""
    def __init__(
        self,
        model_path: str = "Efficient-Large-Model/NVILA-15B",
        conv_mode: str = "auto",
        max_new_tokens: int = 128,
        devices: Optional[Union[str, List[int]]] = None,
    ):
        """
        Initialize the VILA model.
        
        Args:
            model_path: Path to the model or model name
            conv_mode: Conversation mode for the model
            max_new_tokens: Maximum number of tokens to generate
            devices: Device(s) to load the model on
            use_flash_attention: Whether to use flash attention (may not be supported on all GPUs)
        """
        self.model_path = model_path
        self.conv_mode = conv_mode
        self.max_new_tokens = max_new_tokens
        
        # Load model
        self.model = llava.load(model_path, devices=devices)
        
        # Set conversation mode
        clib.default_conversation = clib.conv_templates[conv_mode].copy()
    
    def generate(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[Union[str, List[str], Image.Image, List[Image.Image]]] = None,
        response_format: Optional[ResponseFormat] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate response from the model.
        
        Args:
            system_prompt: System prompt to set model behavior
            user_prompt: User prompt/question
            input_images: Path(s) to input image(s) or PIL.Image(s)
            response_format: Response format (e.g., JSON)
            max_new_tokens: Override default max_new_tokens
            
        Returns:
            Generated text response
        """
        # Prepare the final prompt
        final_prompt = []
        
        # Add media inputs
        if input_images:
            processed_images = self._process_images(input_images)
            for img in processed_images:
                final_prompt.append(img)
            
        
        # Construct text prompt
        text_prompt = ""
        if system_prompt:
            text_prompt += f"{system_prompt}\n\n"
        if user_prompt:
            text_prompt += user_prompt
        
        if text_prompt:
            final_prompt.append(text_prompt)
        
        # Prepare generation config
        generation_config = self.model.default_generation_config
        generation_config.max_new_tokens = max_new_tokens or self.max_new_tokens
        generation_config.do_sample = do_sample
        if temperature is not None:
            generation_config.temperature = temperature
        
        # Generate response
        response = self.model.generate_content(
            final_prompt,
            response_format=response_format,
            generation_config=generation_config
        )
        
        return response
    
    def generate_json(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[Union[str, List[str]]] = None,
        json_schema: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate JSON-formatted response.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            input_images: Input image path(s)
            json_schema: JSON schema for structured output
            **kwargs: Additional arguments for generate()
            
        Returns:
            JSON-formatted response
        """
        if json_schema:
            response_format = ResponseFormat(
                type="json_schema",
                json_schema=JsonSchemaResponseFormat(schema=json_schema)
            )
        else:
            response_format = ResponseFormat(type="json_object")
        
        return self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_images=input_images,
            response_format=response_format,
            **kwargs
        )
    
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
                    processed.append(Image.open(img))
                elif os.path.exists(img):
                    processed.append(Image.open(img))
                else:
                    print(f"Warning: Image not found: {img}")
            elif isinstance(img, Image.Image):
                # PIL.Image object
                processed.append(img)
            else:
                print(f"Warning: Unsupported image type: {type(img)}")
        
        return processed
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
