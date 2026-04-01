from typing import List, Optional, Union, Tuple
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL35Model:
    """Wrapper class for InternVL3.5 model inference."""
    
    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL3_5-8B",
        max_new_tokens: int = 1024,
        input_size: int = 448,
        max_num_tiles: int = 12,
        use_flash_attn: bool = True,
        load_in_8bit: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the InternVL3.5 model.
        
        Args:
            model_path: Path to the pretrained model
            max_new_tokens: Maximum number of tokens to generate
            input_size: Input image size (default: 448)
            max_num_tiles: Maximum number of image tiles for dynamic preprocessing
            use_flash_attn: Whether to use flash attention
            load_in_8bit: Whether to load model in 8-bit mode
            torch_dtype: Torch data type for model weights
            device: Device to run inference on
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.input_size = input_size
        self.max_num_tiles = max_num_tiles
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading InternVL3.5 model from {model_path}...")
        
        # Load model
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Build image transform
        self.transform = self._build_transform(input_size)
        
        print(f"Model loaded successfully on {device}")
    
    def _build_transform(self, input_size: int):
        """Build image transformation pipeline."""
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def _find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: list,
        width: int,
        height: int,
        image_size: int
    ) -> Tuple[int, int]:
        """Find the closest aspect ratio from target ratios."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        return best_ratio
    
    def _dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = True
    ) -> List[Image.Image]:
        """
        Dynamically preprocess image into tiles based on aspect ratio.
        
        Args:
            image: Input PIL Image
            min_num: Minimum number of tiles
            max_num: Maximum number of tiles
            image_size: Size of each tile
            use_thumbnail: Whether to add a thumbnail image
            
        Returns:
            List of preprocessed image tiles
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        # Calculate target aspect ratios
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        # Find closest aspect ratio
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        # Calculate target dimensions
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        # Resize image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        # Split into tiles
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        assert len(processed_images) == blocks
        
        # Add thumbnail if requested
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def _load_image(
        self,
        image: Union[str, Image.Image],
        input_size: Optional[int] = None,
        max_num: Optional[int] = None
    ) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image: Image path or PIL.Image
            input_size: Override default input size
            max_num: Override default max number of tiles
            
        Returns:
            Preprocessed image tensor
        """
        input_size = input_size or self.input_size
        max_num = max_num or self.max_num_tiles
        
        # Load image if path is provided
        if isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                from requests import get
                from io import BytesIO
                response = get(image)
                image = Image.open(BytesIO(response.content)).convert('RGB')
            else:
                image = Image.open(image).convert('RGB')
        
        # Dynamic preprocessing
        images = self._dynamic_preprocess(
            image,
            image_size=input_size,
            use_thumbnail=True,
            max_num=max_num
        )
        
        # Transform images
        pixel_values = [self.transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        
        return pixel_values
    
    def _process_images(
        self,
        input_images: Optional[Union[str, List[str], Image.Image, List[Image.Image]]]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Process multiple images and return combined tensor with patch counts.
        
        Args:
            input_images: Single or list of image paths, URLs, or PIL.Image objects
            
        Returns:
            Tuple of (combined pixel_values tensor, list of patch counts per image)
        """
        if input_images is None:
            return None, None
        
        # Convert single image to list
        if isinstance(input_images, (str, Image.Image)):
            input_images = [input_images]
        
        pixel_values_list = []
        num_patches_list = []
        
        for img in input_images:
            try:
                pixel_values = self._load_image(img)
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))
            except Exception as e:
                print(f"Warning: Failed to process image {img}: {e}")
        
        if not pixel_values_list:
            return None, None
        
        # Combine all images
        pixel_values = torch.cat(pixel_values_list, dim=0)
        
        return pixel_values, num_patches_list
    
    def generate(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: str = "",
        input_images: Optional[Union[str, List[str], Image.Image, List[Image.Image]]] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        history: Optional[List] = None,
        return_history: bool = False,
    ) -> Union[str, Tuple[str, List]]:
        """
        Generate response from the model.
        
        Args:
            system_prompt: System prompt (prepended to user prompt)
            user_prompt: User prompt/question
            input_images: Path(s) or URL(s) to input image(s), or PIL.Image(s)
            max_new_tokens: Override default max_new_tokens
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            history: Conversation history
            return_history: Whether to return updated history
            
        Returns:
            Generated text response, optionally with updated history
        """
        # Build generation config
        generation_config = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": 151645,
        }
        if temperature is not None and do_sample:
            generation_config["temperature"] = temperature
        
        # Combine system and user prompts
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        # Process images
        pixel_values, num_patches_list = None, None
        if input_images:
            pixel_values, num_patches_list = self._process_images(input_images)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.torch_dtype).to(self.device)
                # Add image tokens
                if len(num_patches_list) == 1:
                    full_prompt += "<image>\n"
                else:
                    for i in range(len(num_patches_list)):
                        full_prompt += f"Image-{i+1}: <image>\n"
        
        full_prompt += user_prompt
        
        # Generate response
        try:
            if return_history:
                response, new_history = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=history,
                    return_history=True
                )
                return response, new_history
            else:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_prompt,
                    generation_config,
                    num_patches_list=num_patches_list,
                    history=history,
                    return_history=False
                )
                return response
        except Exception as e:
            print(f"Error during generation: {e}")
            if return_history:
                return "", history
            return ""
    
    def clear_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
