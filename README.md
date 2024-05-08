# Introduction
# Installation
For Installation, please follow the explanation of the upstream repository [joeynmt](https://github.com/joeynmt/joeynmt/tree/main)
# Let it Run
The original joeynmt has three modes, train, test and translate. In the upstream repository, ([joeynmt](https://github.com/joeynmt/joeynmt/tree/main)),
there is an explanation, how one can let each mode run. Appart from the Vanilla model, the models in this repository have the addidiotnal mode idiomtagging, which works
like translate, but you will not get a file with translations, but a file with logits and predictions of the model's idiom-tagging module.
Before letting a model run, no matter in which mode, make sure to execute the following unix command:

cp -r joeynmt\[MODELNAME\]   joeynmt
# Data source
## Parallel Corpus for Translation 
## Idiom labels for idiom-tagging
