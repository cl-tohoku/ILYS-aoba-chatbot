import math
from argparse import Namespace
from collections import namedtuple
from typing import Iterable, Tuple, List

import numpy as np
import torch
from fairseq import checkpoint_utils, tasks, utils
from fairseq.data import encoders

from .tokenizer import SpmTokenizer

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, 'symbols_to_strip_from_output'):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


class NeuralDialogueModel:
    def __init__(self, args):
        self.args = args
        self._load_args(self.args)
        self.bpe = encoders.build_bpe(self.args)
        self.task = tasks.setup_task(self.args)
        self.tokenizer = SpmTokenizer(self.args.spm)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)
        self.models = self._load_models(self.args)
        self.max_positions = self._load_max_positions(self.models, self.task)
        self.generator = self.task.build_generator(self.models, self.args)

    def __call__(self, contexts: List[str]) -> List[str]:
        """
        Args:
            contexts: List of utterances
        Return:
            responses: Responses created by the model
        """
        if isinstance(contexts, str):
            contexts = [contexts]

        # Create context as input
        tokenized_contexts = [self.tokenizer.encode(utt) for utt in contexts]
        if self.bpe is not None:
            tokenized_contexts = [self.bpe.encode(utt) for utt in contexts]
        context_as_input = ' <s> '.join(tokenized_contexts)
        inputs = [context_as_input]

        # decode
        responses = [text for s, text in self.decode(inputs=inputs)]

        return responses

    def create_batches(self, inputs: List[str]) -> Iterable[Batch]:
        """Create batch as input"""
        tokens = [self.task.source_dictionary.encode_line(line, add_if_not_exist=False).long() for line in inputs]
        lengths = [t.numel() for t in tokens]
        itr = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=self.args.max_tokens,
            max_sentences=self.args.max_sentences,
            max_positions=self.max_positions,
        ).next_epoch_itr(shuffle=False)

        for batch in itr:
            yield Batch(
                ids=batch['id'],
                src_tokens=batch['net_input']['src_tokens'],
                src_lengths=batch['net_input']['src_lengths'],
            )

    def decode(self, inputs: List[str]) -> List[Tuple[float, str]]:
        """
        Args:
            inputs: List of sequences as input to the model
        Returns:
            responses: The list of (score, response) which is created by the model
        """
        results = []
        for batch in self.create_batches(inputs):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if self.args.use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            sample = {'net_input': {'src_tokens': src_tokens, 'src_lengths': src_lengths}}
            translations = self.task.inference_step(self.generator, self.models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.task.target_dictionary.pad())
                results.append((id, src_tokens_i, hypos))

        # sort output to match input order
        responses = []
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if self.task.source_dictionary is not None:
                src_str = self.task.source_dictionary.string(src_tokens, self.args.remove_bpe)

            # Process top predictions
            for hypo in hypos[:min(len(hypos), self.args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.align_dict,
                    tgt_dict=self.task.target_dictionary,
                    remove_bpe=self.args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
                )

                if self.bpe is not None:
                    hypo_str = self.bpe.decode(hypo_str)
                detok_hypo_str = self.tokenizer.decode(hypo_str)

                score = hypo['score'] / math.log(2)  # convert to base 2
                responses.append((float(score), detok_hypo_str))

        return responses

    @staticmethod
    def _load_args(args: Namespace) -> None:
        # Settings
        utils.import_user_module(args)
        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.max_sentences is None:
            args.max_sentences = 1
        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        # Fix seed for stochastic decoding
        if args.seed is not None and not args.no_seed_provided:
            np.random.seed(args.seed)
            utils.set_torch_seed(args.seed)

        # Fix and add args option
        args.use_cuda = True if torch.cuda.is_available() and not args.cpu else False

    @staticmethod
    def _load_max_positions(models, task):
        max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        )

        return max_positions

    def _load_models(self, args):
        state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
        state["args"].data = args.data
        task = tasks.setup_task(state["args"])
        model = task.build_model(state["args"])
        model.load_state_dict(state["model"], strict=True, args=state["args"])
        model.make_generation_fast_()
        if args.fp16:
            model.half()
        if args.use_cuda:
            model.cuda()

        return [model]
