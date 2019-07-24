# Introduction

This repository contains scripts for experiments in text modeling with
[agglomerative attention](https://arxiv.org/abs/1907.06607).

# Examples

```
python -m flowws.run -m agglom_attention_flowws \
       Text8 --sequence-length 128 --epoch-scaling-factor .1 \
       GPTModel --width 64 --depth 5 --use-agglomeration True --num-heads 8 \
       Run --early-stopping 8 --metrics bpc --epochs 256

python -m flowws.run -m agglom_attention_flowws \
       WikiText2 --sequence-length 128 \
       GPTModel --width 128 --depth 5 --use-agglomeration True --num-heads 8 \
       Run --early-stopping 8 --metrics perplexity --epochs 256
```
