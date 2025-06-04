import os

import torch
import transformers
from transformers import AutoTokenizer
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM

os.environ["TQDM_DISABLE"] = "1"

# Code from https://github.com/state-spaces/mamba/tree/main


class BaseEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained=None,
        max_length=2048,
        batch_size=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        **kwargs,
    ):
        # Initialize LM parent class
        LM.__init__(self)
        
        # Parameters
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self.torch_dtype = dtype

        # Required HFLM attributes
        self.backend = "causal"
        self.trust_remote_code = True

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        return self._max_length

    def get_model_info(self):
        """
        Returns information about the model
        """
        return {
            "model_name": self.model_name,
            "model_revision": self.revision,
            "batch_size": self.batch_size,
            "device": str(self._device),
        }

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L896-L921
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        
        # Generate to max_length and truncate if needed
        return self._model.generate(
            input_ids=context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs,
        )


@register_model("phi-mamba")
class PhiMambaEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="goombalab/Phi-Mamba", **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        self.model_name = pretrained
        self._model = LMHeadModel.from_pretrained(self.model_name, strict=True)
        self._model.to(self._device).to(self.torch_dtype).eval()
        print(self._model)


@register_model("hybrid-phi-mamba")
class HybridPhiMambaEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="goombalab/Hybrid-Phi-Mamba", **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        self.model_name = pretrained
        self._model = LMHeadModel.from_pretrained(
            self.model_name,
            attn_type="flash_attention_2" if torch.cuda.is_available() else "eager",
            strict=True,
        )
        self._model.to(self._device).to(self.torch_dtype).eval()
        print(self._model)


@register_model("phi")
class PhiEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="microsoft/phi-1_5", **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        self.model_name = pretrained
        self._model = PhiForCausalLM.from_pretrained(self.model_name, strict=True)
        self._model.to(self._device).to(self.torch_dtype).eval()
        print(self._model)


if __name__ == "__main__":
    cli_evaluate()
