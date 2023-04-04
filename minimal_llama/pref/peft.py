from dataclasses import dataclass

PEFT_PREFIX = "prefix"
NO_PEFT = "nothing"


@dataclass
class PeftConfig:
    peft_mode: str
    num_prefix_tokens: int = None
    lora_rank: int = None
    lora_alpha: int = 16

    def check(self):
        assert self.peft_mode in (
            PEFT_PREFIX, NO_PEFT,
        )
