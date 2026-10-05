# JR1 execution environment

The campaign uses the host-provided `/usr/bin/python3` rather than modifying
shared packages. At launch the execution owner records the installed PyTorch,
Transformers and Datasets versions in its private environment receipt.

Run each GPU command with one explicit visible device and campaign-local
caches/state:

```bash
export CUDA_VISIBLE_DEVICES=0
export JR1_STATE_ROOT=/path/to/private/state
python3 experiments/03_qwen_joint_reasoning_retention/expt_v1/jr1_campaign.py test
python3 experiments/03_qwen_joint_reasoning_retention/expt_v1/test_jr1_campaign.py
```

Public model and dataset caches, checkpoints, raw generations and runtime
receipts belong under the private state root. Only compact manifests, frozen
protocols and verified summaries are committed.
