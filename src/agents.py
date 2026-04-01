from typing import Optional

MODELS = {
    "gemma": "google/gemma-3-4b-it",
    # "gemma12b": "google/gemma-3-12b-it",
    "internvl": "OpenGVLab/InternVL3_5-8B",
    # "internvl14b": "OpenGVLab/InternVL3_5-14B",
    "llavaonevision": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "vila": "Efficient-Large-Model/NVILA-8B",
    # "vila15b": "Efficient-Large-Model/NVILA-15B",
}

class VLAgent:
    def __init__(self,
                 model_name: str,
                 system_prompt: str = "",
                 user_prompt: str = "",
                 max_new_tokens: int = 128):
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODELS.keys())}.")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self._load_model()
    
    def _load_model(self):
        if self.model_name == "gemma" or self.model_name == "gemma12b":
            from src.gemma3 import Gemma3Model
            self.model = Gemma3Model(model_path=MODELS[self.model_name],
                                     max_new_tokens=self.max_new_tokens)
        elif self.model_name == "internvl" or self.model_name == "internvl14b":
            from src.internvl35 import InternVL35Model
            self.model = InternVL35Model(model_path=MODELS[self.model_name],
                                         max_new_tokens=self.max_new_tokens)
        # elif self.model_name == "llavanext":
        #     from src.llavanext import LLaVANextModel
        #     self.model = LLaVANextModel(model_path=MODELS[self.model_name],
        #                                 max_new_tokens=self.max_new_tokens)
        elif self.model_name == "llavaonevision":
            from src.llavaonevision import LLaVAOneVisionModel
            self.model = LLaVAOneVisionModel(model_path=MODELS[self.model_name],
                                             max_new_tokens=self.max_new_tokens)
        elif self.model_name == "qwen":
            from src.qwen3vl import Qwen3VLModel
            self.model = Qwen3VLModel(model_path=MODELS[self.model_name],
                                      max_new_tokens=self.max_new_tokens)
        elif self.model_name == "vila" or self.model_name == "vila15b":
            from src.vila import VILAModel
            self.model = VILAModel(model_path=MODELS[self.model_name],
                                   max_new_tokens=self.max_new_tokens)
        else:
            raise ValueError(f"Model '{self.model_name}' is not implemented.")

    def chat(self,
        input_images,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        do_sample: bool = True,
        temperature: Optional[float] = None,
        **kwargs,
        ) -> str:
        self.system_prompt = system_prompt if system_prompt else self.system_prompt
        self.user_prompt = user_prompt if user_prompt else self.user_prompt
        return self.model.generate(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            input_images=input_images,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            **kwargs
        )