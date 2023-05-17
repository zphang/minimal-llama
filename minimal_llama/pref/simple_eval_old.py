import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union


from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import pandas as pd


def score_eval(predictions, dataset, task_name=None):
    hash_dict = {}
    for pred_score, example in zip(predictions, dataset):
        key = str(tuple(example["input_ids"]))
        gold = str(tuple(example["labels"]))
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


class ScoreEvalTrainer(Seq2SeqTrainer):

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                full_input_ids = torch.cat([
                    inputs["input_ids"],
                    inputs["labels"],
                ], dim=1).cuda().clamp(min=0)
                full_lm_logits = model(
                    input_ids=full_input_ids,
                )
                label_lm_logits = full_lm_logits[:, inputs["input_ids"].shape[1]:]
                orig_shape = inputs["labels"].shape
                flat_shape = inputs["labels"].numel()
                loss = F.cross_entropy(
                    label_lm_logits.reshape(flat_shape, -1),
                    inputs["labels"].cuda().reshape(flat_shape),
                    ignore_index=-100,
                    reduction="none",
                )
                loss_scores = loss.reshape(orig_shape).sum(-1)
            loss = loss_scores.mean().detach()

        labels = inputs["labels"]
        return loss, loss_scores, labels



