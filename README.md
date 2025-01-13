
> **_NOTE:_**  This repository is a work in progress. Changes and updates may occur as the project evolves.

# ARC-TTT
Simplified and Stripped-Down ARC-Test-Time-Training (ARC-TTT). This implementation uses [Hugging Face](https://huggingface.co/) and [unsloth](https://unsloth.ai/), based on the original [MARC](https://github.com/ekinakyurek/marc/tree/main) repository, which used [torchtune](https://github.com/pytorch/torchtune).
Credits to [MARC](https://github.com/ekinakyurek/marc/tree/main).

## Env Setup:

Create your conda environment for training with unsloth as explained [here](https://github.com/unslothai/unsloth). 


## Run Arc TTT

```bash
python ttt_unsloth.py
```

Original implementation uses a fine-tuned base model. In this repo we fine-tune task adapter from scratch of a regular pre-trained model that is not yet fine-tuned on the ARC tasks.