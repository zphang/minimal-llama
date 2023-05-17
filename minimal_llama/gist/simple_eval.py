import os
import pandas as pd
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
import transformers

import minimal_llama.gist.data.p3 as p3_datasets
import minimal_llama.gist.llama_gist as llama_gist
import proj_shared.assets_utils as assets_utils
import proj_shared.io_utils as io_utils


@dataclass
class EvalArguments:
    dataset_path: str = field()
    hf_path: str = field()
    model_load_path: str = field()
    model_name: str = field(default="7b")

    # p3_specific
    eval_mode: str = field(metadata={"help": "(score|in_domain|score,in_domain)"}, default="score")
    p3_subset_name: str = field(default="t0_short")
    p3_eval_quick: bool = field(default=False)
    add_answer_indicator: bool = field(default=False)

    save_fol: str = field(
        default=None,
        metadata={"help": "Path to save to."},
    )
    rng_seed: int = field(
        metadata={"help": "Seed for few-shot sampling"},
        default=1234,
    )
    do_test: bool = field(
        default=False,
        metadata={"help": ""},
    )


@dataclass
class GistArguments:
    data_mode: str = field(default="multigist")
    max_num_examples: int = field(default=32)
    num_gist_tokens: int = field(default=8)
    max_sequence_length: int = field(default=512)


def parse_args(parser: HfArgumentParser):
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        arg_objs = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        arg_objs = parser.parse_args_into_dataclasses()
    return arg_objs


class ScoreEvalTrainer(Trainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        labels = inputs["labels"]
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            with self.compute_loss_context_manager():
                lm_logits = model(input_ids=input_ids, attention_mask=inputs["attention_mask"])
                orig_shape = inputs["labels"].shape
                flat_shape = inputs["labels"].numel()
                loss = F.cross_entropy(
                    lm_logits.reshape(flat_shape, -1),
                    labels.reshape(flat_shape),
                    ignore_index=-100,
                    reduction="none",
                )
                loss_scores = loss.reshape(orig_shape).sum(-1)
            loss = loss_scores.mean().detach()

        return loss, loss_scores, labels


class HyperSeq2SeqTrainer(Seq2SeqTrainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # noinspection PyUnresolvedReferences
        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )
        inputs = self._prepare_inputs(inputs)
        # TODO: This function hasn't been tested at all!
        with torch.inference_mode():
            generated_tokens = ...

        return None, generated_tokens, inputs["labels"]


def score_pred(model_out, dataset, tokenizer):
    output_list = []
    for i in range(len(dataset)):
        output_list.append({
            "gold": tokenizer.decode(p3_datasets.clean_label(model_out.label_ids[i][0])),
            "pred": tokenizer.decode(p3_datasets.clean_pred(model_out.predictions[i][0])),
        })
    df = pd.DataFrame(output_list)
    # Only accuracy for now
    # noinspection PyUnresolvedReferences
    accuracy = float((df["gold"] == df["pred"]).mean())
    return accuracy


def score_eval(predictions, dataset, tokenizer, task_name=None):
    hash_dict = {}
    for pred_score, example in zip(predictions, dataset):
        gold = tokenizer.decode(p3_datasets.clean_label(example["labels"].tolist()))
        key = example["key"]
        if key not in hash_dict:
            hash_dict[key] = {"scores": {}, "is_correct": None}
        if task_name and not task_name.startswith("super_glue_wsc.fixed_"):
            # super_glue_wsc.fixed_ have one set of repeated examples
            assert gold not in hash_dict[key]["scores"]
        hash_dict[key]["scores"][gold] = pred_score
        if example["is_correct"]:
            if task_name and not task_name.startswith("super_glue_wsc.fixed_"):
                assert hash_dict[key]["is_correct"] is None
            hash_dict[key]["is_correct"] = gold

    num_correct = 0
    for row in hash_dict.values():
        # Lower loss-score is better
        prediction = pd.Series(row["scores"]).idxmin()
        if prediction == row["is_correct"]:
            num_correct += 1
    accuracy = num_correct / len(hash_dict)
    return accuracy


def evaluate_with_score(model, tokenizer,
                        gist_args: GistArguments,
                        eval_args: EvalArguments,
                        training_args):
    trainer = ScoreEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        tokenizer=tokenizer,
        data_collator=p3_datasets.data_collator,
    )

    if training_args.predict_with_generate:
        filename = "t0_eval_prelim.json"
    else:
        if eval_args.p3_eval_quick:
            filename = "t0_eval_score_prelim_quick.json"
        else:
            filename = "t0_eval_score_prelim.json"
    t0_eval_metadata = io_utils.read_json(assets_utils.get_assets_path("subsets", filename))

    all_results = []
    for i, row in enumerate(t0_eval_metadata):
        val_ds = p3_datasets.P3FewshotHyperValidationDataset(
            dataset_name=row["name"],
            base_path=eval_args.dataset_path,
            rng_seed=eval_args.rng_seed + i,
            num_gist_tokens=gist_args.num_gist_tokens,
            mode=gist_args.data_mode,
            full_sequence_length=gist_args.max_sequence_length,
            add_special_tokens=True,
            add_answer_indicator=eval_args.add_answer_indicator,
        )

        model_out = trainer.predict(val_ds)

        if training_args.predict_with_generate:
            accuracy = score_pred(
                model_out=model_out,
                dataset=val_ds,
                tokenizer=tokenizer,
            )
        else:
            accuracy = score_eval(
                predictions=model_out.predictions,
                dataset=val_ds,
                tokenizer=tokenizer,
                task_name=row["name"],
            )

        all_results.append({
            "name": row["name"],
            "base_dataset": row["base_dataset"],
            "score": accuracy,
        })
        if trainer.is_world_process_zero():
            print(f"[{i}/{len(t0_eval_metadata)}]", row["name"], accuracy)
    return trainer, all_results


def evaluate_in_domain(model, tokenizer, eval_args: EvalArguments, training_args):
    training_args.predict_with_generate = True
    trainer = HyperSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        tokenizer=tokenizer,
        data_collator=p3_datasets.data_collator,
    )
    all_results = []
    total = len(in_domain.TASK_EVALUATOR_MAPPING)
    for i, (dataset_name, evaluator) in enumerate(in_domain.TASK_EVALUATOR_MAPPING.items()):
        val_ds = p3_datasets.P3FewshotHyperValidationDataset(
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            base_path=task_args.data_base_path,
            rng_seed=task_args.rng_seed + i,
            num_actual_examples=1,
            num_hyper_examples=task_args.num_hyper_examples,
            max_hyper_input_length=task_args.max_hyper_input_length,
            max_downstream_input_length=task_args.max_downstream_input_length,
            max_downstream_target_length=task_args.max_downstream_target_length,
            hyper_input_format="concatenated",
            actual_examples_mode="multiple",
            do_test=task_args.do_test,
            add_eos_to_input=task_args.add_eos_to_input,
        )
        model_out = trainer.predict(val_ds)
        preds = []
        labels = []
        for example_i in range(len(model_out.predictions)):
            preds.append(tokenizer.decode(hyper_p3_datasets.clean_pred(
                model_out.predictions[example_i][0, :]
            )))
            labels.append(tokenizer.decode(hyper_p3_datasets.clean_label(
                model_out.label_ids[example_i][0, :]
            )))
        result = in_domain.to_floats(evaluator.evaluate(
            preds=preds,
            labels=labels
        ))
        all_results.append({
            "name": dataset_name,
            "base_dataset": in_domain.TASK_BASE_DATASET_MAPPING[dataset_name],
            "score": float(pd.Series(result).mean()),
            "metrics": result,
        })
        if trainer.is_world_process_zero():
            print(f"[{i}/{total}]", dataset_name, result)
    return trainer, all_results


def save_results(trainer, save_fol, filename, all_results, do_test):
    if do_test:
        filename = "TEST_" + filename
    if trainer.is_world_process_zero():
        os.makedirs(save_fol, exist_ok=True)
        save_path = os.path.join(save_fol, filename)
        io_utils.write_json(all_results, save_path)
        print(f"Saved to {save_path}")


def main():
    eval_args, gist_args, training_args = parse_args(HfArgumentParser((
        EvalArguments,
        GistArguments,
        Seq2SeqTrainingArguments,
    )))
    training_args.remove_unused_columns = False

    tokenizer = transformers.LlamaTokenizer.from_pretrained(eval_args.hf_path)
    if eval_args.save_fol is not None:
        save_fol = eval_args.save_fol
    else:
        save_fol = os.path.split(eval_args.model_path)[0]

    # Load model
    # Model Setup (TODO: move)
    model = llama_gist.create_model(
        model_name=eval_args.model_name,
        hf_path=eval_args.hf_path,
        num_gist_tokens=gist_args.num_gist_tokens,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )
    model.eval_args = eval_args
    model.load_state_dict(torch.load(eval_args.model_load_path, map_location="cpu"))
    model = model.cuda().eval()

    for eval_mode in eval_args.eval_mode.split(","):
        if eval_mode == "score":
            trainer, all_results = evaluate_with_score(
                model=model,
                tokenizer=tokenizer,
                eval_args=eval_args,
                gist_args=gist_args,
                training_args=training_args,
            )
            # noinspection PyTypeChecker
            save_results(trainer=trainer, save_fol=save_fol, do_test=eval_args.do_test,
                         filename="eval_score_metrics.json", all_results=all_results)

        elif eval_mode == "in_domain":
            trainer, all_results = evaluate_in_domain(
                model=model,
                tokenizer=tokenizer,
                eval_args=eval_args,
                training_args=training_args,
            )
            # noinspection PyTypeChecker
            save_results(trainer=trainer, save_fol=save_fol, do_test=eval_args.do_test,
                         filename="in_domain_metrics.json", all_results=all_results)
        else:
            raise KeyError(eval_args.eval_mode)


if __name__ == "__main__":
    main()
