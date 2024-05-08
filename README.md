# Introduction
# Installation
For Installation, please follow the explanation of the upstream repository [joeynmt](https://github.com/joeynmt/joeynmt/tree/main)
# Let it Run
The original joeynmt has three modes, train, test and translate. In the upstream repository, ([joeynmt](https://github.com/joeynmt/joeynmt/tree/main)),
there is an explanation, how one can let each mode run. Appart from the Vanilla model, the models in this repository have the addidiotnal mode idiomtagging, which works
like translate, but you will not get a file with translations, but a file with logits and predictions of the model's idiom-tagging module.
Before letting a model run, no matter in which mode, make sure to execute the following unix command:

cp -r joeynmtMODELNAME/ joeynmt/

The name of a configfile contains the name of the model it belongs to. If you want to train the MTL-model that has a simple linear layer idiom-tagging module
and does not use its output to modify any attention patterns you has to follow these steps:

cp -r joeynmtLinear/ joeynmt/
nohup python -m joeynmt train configs/

# Data source
## Parallel Corpus for Translation 
## Idiom labels for idiom-tagging
