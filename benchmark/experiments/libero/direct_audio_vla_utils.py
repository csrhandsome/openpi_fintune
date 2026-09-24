import torch
import numpy as np
import PIL.Image
from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration, GenerationConfig
from typing import Optional, Union, Dict, Any
# from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info
from huggingface_hub import hf_hub_download
import json
from pathlib import Path

if not hasattr(torch, "get_default_device"):
    def _get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.get_default_device = _get_default_device

import os

SPOKEN_INSTRUCTION_PROMPT = "Follow the spoken instruction."
    

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action

class DirectAudioVLA:
    
    # Define action token range and normalization bounds as class attributes
    # These are specific to the model's vocabulary and task
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712


    def __init__(
        self,
        model_path: str ="MODEL_PATH", 
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16 # Make dtype configurable
    ):
        """
        Initializes the QwenRoboInference model and processors.

        Args:
            model_path (str): Hugging Face model ID or local path for the main
                            Qwen 2.5 Omni model and processor.
            device (Optional[str]): The device to use for inference (e.g., "cuda:0", "cpu").
                                     If None, automatically detects CUDA or falls back to CPU.
            torch_dtype (torch.dtype): The data type to use for the model.
                                       Defaults to torch.bfloat16.
        Raises:
            RuntimeError: If models or processors fail to load, or device is unavailable.
        """
        # --- Device Setup ---

        self.libero_keys= {'libero_object':[np.array( [
        -0.5383928418159485,
        -0.8758928775787354,
        -0.9375,
        -0.06964285671710968,
        -0.11678571254014969,
        -0.15964286029338837,
        0.0
      ]),np.array([
        0.8464285731315613,
        0.84375,
        0.9375,
        0.08142857253551483,
        0.14892856776714325,
        0.0867857113480568,
        1.0
      ])],'libero_object_min':[np.array([-0.8839286 , -0.9375    , -0.9375    , -0.15000001, -0.29035714,
        -0.32892856, -1.        ]),np.array([0.9375    , 0.89196426, 0.9375    , 0.17678571, 0.35035715,
        0.18107143, 1.        ])],
      'libero_10':[np.array([
        -0.6348214149475098,
        -0.7741071581840515,
        -0.7633928656578064,
        -0.09749999642372131,
        -0.14819999992847435,
        -0.2742857038974762,
        0.0
      ]),np.array([
        0.7714285850524902,
        0.8464285731315613,
        0.9375,
        0.13928571343421936,
        0.15964286029338837,
        0.3246428668498993,
        1.0
      ])],'libero_90':[np.array([
        -0.6348214149475098,
        -0.7741071581840515,
        -0.7633928656578064,
        -0.09749999642372131,
        -0.14819999992847435,
        -0.2742857038974762,
        0.0
      ]),np.array([
        0.7714285850524902,
        0.8464285731315613,
        0.9375,
        0.13928571343421936,
        0.15964286029338837,
        0.3246428668498993,
        1.0
      ])],'libero_goal':[np.array([-0.8785714507102966,
        -0.7553571462631226,
        -0.9375,
        -0.1510714292526245,
        -0.1639285683631897,
        -0.13777500048279764,
        0.0])
       ,np.array([0.9375,
        0.9107142686843872,
        0.9375,
        0.20357142388820648,
        0.26357144117355347,
        0.375,
        1.0])],'libero_spatial':[np.array([
        -0.7454732114076613,
        -0.6616071462631226,
        -0.9375,
        -0.1071428582072258,
        -0.20678570866584778,
        -0.1842857152223587,
        0.0
      ]),np.array([
        0.9375,
        0.8758928775787354,
        0.9321428537368774,
        0.1039285734295845,
        0.17678570747375488,
        0.14571428298950195,
        1.0
      ])],
      'libero_spatial_min':[np.array([-0.9375    , -0.9375    , -0.9375    , -0.1875    , -0.36750001,
        -0.36000001, -1]),
        np.array([0.9375    , 0.9375    , 0.9375    , 0.19714285, 0.33642858,
        0.375     , 1.        ])]}


        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")

        if self.device.startswith("cuda"):
             if not torch.cuda.is_available():
                  raise RuntimeError(f"CUDA is not available, but device '{self.device}' was specified.")
             gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
             if gpu_id >= torch.cuda.device_count():
                  raise RuntimeError(f"CUDA device {gpu_id} not available. Only {torch.cuda.device_count()} devices found.")

        # --- Load Fast Tokenizer ---
        try:
            print(f"Loading fast tokenizer from: physical-intelligence/fast")
            self.fast_tokenizer = AutoProcessor.from_pretrained(
                "physical-intelligence/fast", trust_remote_code=True
            )
            # Ensure required attributes are set/exist
           
            self.fast_tokenizer.action_dim = 7 # Set default if not in config
            print("Setting action_dim  to 7.")
           
            self.fast_tokenizer.time_horizon = 1 # Set default if not in config
            print("Setting time horizon to 1.")

        except Exception as e:
            raise RuntimeError(
                f"Error loading fast tokenizer: {e}. "
            )

        # --- Load Main Processor ---
        try:
            print(f"Loading main processor from: {model_path}")
            # Assuming the main processor is saved in the same location as the model
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"Error loading main processor from {model_path}: {e}")

        # --- Load Main Model ---
        try:
            print(f"Loading model from: {model_path}")
            self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                # attn_implementation="flash_attention_2", # Comment out this line if there is an error with flash attention
            )
            self.model.to(self.device)
            # Load generation config and set specific parameters
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
            self.model.generation_config.do_sample = False # Ensure deterministic output

            self.model.eval() # Set the model to evaluation mode


            file_path = Path(__file__).with_name("norm_stats.json")

            # Load the JSON file
            with open(file_path, "r") as f:
                norm_stats = json.load(f)
            self.norm_stats = norm_stats
            

        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")

        print("Model and processors loaded successfully.")
    
    @torch.inference_mode()
    def inference(
        self,
        image: np.ndarray,
        task_item: dict,
        unnorm_key: str = None,
        unnormalizer=None,
        conditioning_mode: str = "audio",
    ) -> np.ndarray:
        """Runs direct audio+text LIBERO action inference.

        The direct benchmark intentionally removes multi-turn confirmation.
        Audio mode uses current image + spoken command + a constant text prompt.
        ASR-baseline mode uses current image + degraded-ASR transcript text only.
        """
        if conditioning_mode not in {"audio", "asr_baseline"}:
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")

        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        speech_conv = task_item.get("speech_conv") or []
        if isinstance(speech_conv, str):
            speech_conv = [speech_conv]
        audio_path = task_item.get("audio_path") or (speech_conv[0] if speech_conv else None)
        audio_root_dir = task_item.get("audio_root_dir", "./data")

        content = [
            {
                "type": "image",
                "image": image,
                "resized_height": 224,
                "resized_width": 224,
            }
        ]
        if conditioning_mode == "asr_baseline":
            asr_text = task_item.get("asr_baseline_text") or task_item.get("text_prompt") or ""
            if not asr_text:
                raise ValueError("ASR baseline requires `asr_baseline_text` or `text_prompt` in task_item.")
            content.append({"type": "text", "text": asr_text})
        else:
            if audio_path:
                content.append({"type": "audio", "audio": os.path.join(audio_root_dir, audio_path.lstrip("./"))})
            content.append({"type": "text", "text": SPOKEN_INSTRUCTION_PROMPT})

        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, image_inputs, video_inputs = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        action_token_idx = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        action_token_idx = torch.where(action_token_idx)[0]
        if len(action_token_idx) == 0:
            output_ids = generated_ids[:, inputs["input_ids"].size(1):]
            response = self.processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            raise ValueError(f"Model did not emit action tokens. Decoded response: {response}")

        output_action = self.fast_tokenizer.decode([generated_ids[0][action_token_idx] - self._ACTION_TOKEN_MIN])
        if unnormalizer is not None:
            unnormalized_action = unnormalizer({"action": output_action})
            return unnormalized_action["action"]

        if "libero" in unnorm_key:
            action_low, action_high = self.libero_keys[unnorm_key]
        else:
            action_norm_stats = self.get_action_stats(unnorm_key)
            action_high = np.array(action_norm_stats["q99"])
            action_low = np.array(action_norm_stats["q01"])

        unnorm_actions = 0.5 * (output_action + 1) * (action_high - action_low) + action_low
        return np.array(unnorm_actions[0])

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
    