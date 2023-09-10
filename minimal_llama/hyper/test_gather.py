import accelerate
from accelerate.utils.operations import gather_object


def run():
    # Initialize Accelerate
    # Needed because technically some LoRA weights aren't actually used (final attn-O and MLP)
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print0 = accelerator.on_local_main_process(print)
    local_rank = accelerator.process_index
    print(f"Rank {local_rank}, {device}")
    # x = [f"Rank {local_rank}, {device}"]
    x = [{"x": f"Rank {local_rank}, {device}"}, {"y": f"lol"}]
    print0(gather_object(x))


if __name__ == "__main__":
    run()
