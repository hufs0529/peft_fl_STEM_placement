"""PEFT 래핑 로직 — 연구계획서 Task 1/Task 2와 정확히 일치.

core (Aim 1, Subtask 1.1): LoRA(r=8), QLoRA(NF4 4bit) — 3 FL x 2 PEFT = 6조합.
diagnostic (Aim 2, Subtask 2.1): DoRA — 양자화 없이 LoRA와 동일한 압축률을
갖는 대조군. Task 1에서 상호작용 효과가 가장 컸던 FL 알고리즘과만 짝지어
1회 실행해, 그 효과가 "양자화 노이즈" 때문인지 "압축 자체" 때문인지 분리한다.

Prompt Tuning/Adapter Tuning 등 다른 PEFT 방식은 이번 연구계획서의 범위에
없으므로 포함하지 않는다.
"""

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model


def get_model(config: dict):
    """config['model']['name'], config['peft']['type'] 기준으로 모델 생성."""
    model_name = config["model"]["name"]
    peft_type = config["peft"]["type"]

    if peft_type == "lora":
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        lora_cfg = LoraConfig(
            r=config["peft"]["lora_r"],
            lora_alpha=config["peft"]["lora_alpha"],
            lora_dropout=config["peft"].get("lora_dropout", 0.05),
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_cfg)

    elif peft_type == "qlora":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_cfg, device_map="auto"
        )
        lora_cfg = LoraConfig(
            r=config["peft"]["lora_r"],
            lora_alpha=config["peft"]["lora_alpha"],
            lora_dropout=config["peft"].get("lora_dropout", 0.05),
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        return get_peft_model(model, lora_cfg)

    elif peft_type == "dora":
        # Subtask 2.1 진단 대조군: 양자화 없는 DoRA. Task 1에서 상호작용
        # 효과가 가장 컸던 FL 알고리즘과 짝지어 실행해 "양자화 때문"인지
        # "압축 자체 때문"인지 분리하기 위한 조건.
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        lora_cfg = LoraConfig(
            r=config["peft"]["lora_r"],
            lora_alpha=config["peft"]["lora_alpha"],
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,
        )
        return get_peft_model(model, lora_cfg)

    raise ValueError(f"Unknown peft type: {peft_type} (연구계획서는 lora/qlora/dora만 정의)")
