#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a cybersecurity assistant analysing emails for phishing detection."
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    adapter_type: str
    context_limit: int


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "meta-llama/Llama-3.1-8B-Instruct": ModelSpec(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        family="llama",
        adapter_type="llama",
        context_limit=128000,
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen",
        adapter_type="qwen",
        context_limit=131072,
    ),
}


class BaseModelAdapter(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model_name = spec.model_id
        self.family = spec.family
        self.context_limit = spec.context_limit

        self.tokenizer = None
        self.model = None

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def build_input(self, prompt: str) -> dict:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        pass

    def format_chat_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def count_tokens(self, prompt: str) -> int:
        messages = self.format_chat_messages(prompt)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        return input_ids.shape[1]

    def get_metadata(self) -> dict:
        return {
            "model_name": self.model_name,
            "family": self.family,
            "context_limit": self.context_limit,
        }


class LlamaAdapter(BaseModelAdapter):
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=True,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        self.model.eval()

    def build_input(self, prompt: str) -> dict:
        messages = self.format_chat_messages(prompt)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        input_ids = input_ids.to(self.model.device)

        attention_mask = torch.ones_like(
            input_ids,
            device=self.model.device,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        model_inputs = self.build_input(prompt)
        input_ids = model_inputs["input_ids"]

        terminators = [self.tokenizer.eos_token_id]

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != self.tokenizer.unk_token_id:
            terminators.append(eot_id)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        response_ids = outputs[0][input_ids.shape[-1]:]

        return self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()


class QwenAdapter(BaseModelAdapter):
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(
            self.model_name,
            local_files_only=True,
            trust_remote_code=True,
        )

        config.rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn",
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=config,
            local_files_only=True,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model.eval()

    def build_input(self, prompt: str) -> dict:
        messages = self.format_chat_messages(prompt)

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(
                inputs["input_ids"],
                device=self.model.device,
            )

        return inputs

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        model_inputs = self.build_input(prompt)
        input_ids = model_inputs["input_ids"]

        terminators = [self.tokenizer.eos_token_id]

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        response_ids = outputs[0][input_ids.shape[-1]:]

        return self.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        ).strip()


def get_model_spec(model_id: str) -> ModelSpec:
    if model_id not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unsupported model: {model_id}\n"
            f"Known models: {known}"
        )

    return MODEL_REGISTRY[model_id]


def create_adapter(model_id: str) -> BaseModelAdapter:
    spec = get_model_spec(model_id)

    if spec.adapter_type == "llama":
        return LlamaAdapter(spec)

    if spec.adapter_type == "qwen":
        return QwenAdapter(spec)

    raise ValueError(
        f"Unsupported adapter_type={spec.adapter_type} for model {model_id}"
    )


def list_supported_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())