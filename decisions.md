# Decisions (written by the project owner, not by an AI)

Agents may quote these but must never add to them. Wording below is the owner's account, lightly cleaned up.

## D-01
Decision: Use LoRA fine-tuning with LoRA configurations compared against each other.
Why: The aim is to get better results while adding as few trainable weights as possible. The mentor also suggested it.

## D-02
Decision: Rank sweeps were not run equally for every LoRA setup. balanced_qkvo has more ranks than efficient_qv and all_linear.
Why: Time limits. The owner also says the efficient (q,v) setup did not show good results in other experiments, so it was not extended.

## D-03
Decision: The zero-shot and few-shot baselines use a Qwen-based VLM, and are kept in the project for comparison only.
Why: Llama-based VLM weights could not be found for the zero/few-shot setting. The Qwen VLM was run first and Llama-based models later. The mentor then said comparing Llama and Qwen in zero/few-shot is not right, so the Qwen baseline stays only for comparison's sake.

## D-04
Decision: Include a frozen zero-shot / few-shot baseline.
Why: To see how VLMs perform without training and compare with their performance after fine-tuning.

## D-05
Decision: Use a dataset built by another paper on top of the existing FLAME-3 dataset.
Why: The project goal is a VQA system for wildfires.

## D-06
Decision: Exact Match is considered the most important metric by the owner.
Why: The expected outputs are very limited, and the owner thinks BLEU and ROUGE are not appropriate for this task.

## D-07
Decision: The project is a visual question answering (VQA) system for wildfires.
Why: (stated as the project goal)

## D-08
Decision: Wildfire_Generative_Custom_VLM_Kaggle.ipynb was a trial run only.
Why: Early experiment before the main runs.
