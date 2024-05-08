import logging
import math
import sys
import time
from functools import partial
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from joeynmt.data import load_data
from joeynmt.datasets import StreamDataset, build_dataset
from joeynmt.helpers import (
    check_version,
    expand_reverse_index,
    load_checkpoint,
    load_config,
    make_logger,
    parse_test_args,
    parse_train_args,
    resolve_ckpt_path,
    set_seed,
    store_attention_plots,
    write_list_to_file,
)
from joeynmt.metrics import bleu, chrf, sequence_accuracy, token_accuracy
from joeynmt.model import Model, _DataParallel, build_model
from joeynmt.search import search
from joeynmt.tokenizers import build_tokenizer
from joeynmt.vocabulary import build_vocab

logger = logging.getLogger(__name__)

def idiomtagging(
    cfg_file: str,
    ckpt: str = None,
    output_path: str = None,
) -> None:
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or asks for
    input to translate interactively. Translations and scores are printed to stdout.
    Note: The input sentences don't have to be pre-tokenized.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output file
    """
    cfg = load_config(Path(cfg_file))
    # parse and validate cfg
    model_dir, load_model, device, n_gpu, num_workers, _, fp16 = parse_train_args(
        cfg["training"], mode="prediction")
    test_cfg = cfg["testing"]
    src_cfg = cfg["data"]["src"]
    trg_cfg = cfg["data"]["trg"]

    pkg_version = make_logger(model_dir, mode="translate")  # version string returned
    if "joeynmt_version" in cfg:
        check_version(pkg_version, cfg["joeynmt_version"])

    # when checkpoint is not specified, take latest (best) from model dir
    load_model = load_model if ckpt is None else Path(ckpt)
    ckpt = resolve_ckpt_path(load_model, model_dir)
    

    # read vocabs
    src_vocab, trg_vocab, idiom_vocab = build_vocab(cfg["data"], model_dir=model_dir)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab,idiom_vocab=idiom_vocab)

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, device=device)
    model.load_state_dict(model_checkpoint["model_state"])

    if device.type == "cuda":
        model.to(device)
    model.eval()

    tokenizer = build_tokenizer(cfg["data"])
    sequence_encoder = {
        src_cfg["lang"]: partial(src_vocab.sentences_to_ids, bos=False, eos=True),
        trg_cfg["lang"]: None,
    }
    test_data = build_dataset(
        dataset_type="stream",
        path=None,
        src_lang=src_cfg["lang"],
        trg_lang=trg_cfg["lang"],
        split="test",
        tokenizer=tokenizer,
        sequence_encoder=sequence_encoder,
    )

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    #n_best = test_cfg.get("n_best", 1)
    #beam_size = test_cfg.get("beam_size", 1)
    #return_prob = test_cfg.get("return_prob", "none")
    if not sys.stdin.isatty():  # pylint: disable=too-many-nested-blocks
        # input stream given
        for i, line in enumerate(sys.stdin.readlines()):
            if not line.strip():
                # skip empty lines and print warning
                logger.warning("The sentence in line %d is empty. Skip to load.", i)
                continue
            test_data.set_item(line.rstrip())
        all_hypotheses, tokens = [],[]
        #print(idiom_vocab._stoi)
        for instance in test_data:
            tokens.append(instance)
            batch = test_data.collate_fn([instance])
            batch._make_cuda(device)
            hypothese = model.label_sequence(batch)[0]
            #labels = [idiom_vocab._itos[index] for index in hypothese.tolist()]
            print(hypothese,"\n",instance,"\n")
            all_hypotheses.append(hypothese)
        assert len(all_hypotheses) == len(tokens)#len(test_data) * n_best

        if output_path is not None:
            # write to outputfile if given
            out_file = Path(output_path).expanduser()
            """
            if n_best > 1:
                for n in range(n_best):
                    write_list_to_file(
                        out_file.parent / f"{out_file.stem}-{n}.{out_file.suffix}",
                        [
                            all_hypotheses[i]
                            for i in range(n, len(all_hypotheses), n_best)
                        ],
                    )
            else:
                write_list_to_file(out_file, all_hypotheses)
            """
            
            logger.info("Translations saved to: %s.", out_file)

        else:
            # print to stdout
            for hyp in all_hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        test_cfg["batch_size"] = 1  # CAUTION: this will raise an error if n_gpus > 1
        test_cfg["batch_type"] = "sentence"
        np.set_printoptions(linewidth=sys.maxsize)  # for printing scores in stdout
        while True:
            try:
                src_input = input("\nPlease enter a source sentence:\n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data.set_item(src_input.rstrip())
                hypotheses, tokens, scores = _translate_data(test_data, test_cfg)

                print("JoeyNMT:")
                for i, (hyp, token,
                        score) in enumerate(zip_longest(hypotheses, tokens, scores)):
                    assert hyp is not None, (i, hyp, token, score)
                    print(f"#{i + 1}: {hyp}")
                    if return_prob in ["hyp"]:
                        if beam_size > 1:  # beam search: sequence-level scores
                            print(f"\ttokens: {token}\n\tsequence score: {score[0]}")
                        else:  # greedy: token-level scores
                            assert len(token) == len(score), (token, score)
                            print(f"\ttokens: {token}\n\tscores: {score}")

                # reset cache
                test_data.cache = {}

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
