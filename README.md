# ARC-TTT

Simplified and striped down ARC-Test-Time-Training (ARC-TTT) with huggingface and unsloth.
Originally from [MARC](https://github.com/ekinakyurek/marc/tree/main)


## Env Setup:

Create your conda environment for training with unsloth as explained [here](https://github.com/unslothai/unsloth). 


## Run Arc TTT

```bash
python ttt_unsloth.py
```

Original implementation uses a fine-tuned base model. In this repo we fine-tune task adapter from scratch of a regular pre-trained model that is not yet fine-tuned on the ARC tasks.