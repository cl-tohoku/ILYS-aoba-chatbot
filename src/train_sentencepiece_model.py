# -*- coding: utf-8 -*-

import argparse
from os import path

import sentencepiece
from logzero import logger
from tqdm import tqdm

URL = "https://github.com/google/sentencepiece"

# We have prepared 20 special tokens for fine-tuning and decoding with special tokens.
SPECIAL_TOKENS = ["<ST{id}>".format(id=i) for i in range(1, 21)]


def create_parser():
    parser = argparse.ArgumentParser(description='Training sentencepiece model ({url})'.format(url=URL))
    parser.add_argument('--input', required=True, type=path.abspath, help='Path to input file')
    parser.add_argument('--prefix', required=True, type=str, help='Model prefix')
    parser.add_argument('--output_dir', required=True, type=path.abspath, help='Path to output dir')
    parser.add_argument('--n_vocab', type=int, default=32000, help='Number of vocab')
    parser.add_argument('--coverage', type=float, default=0.9999, help='Character coverage')

    return parser


def count_file_length(file_path: str) -> int:
    with open(file_path) as fi:
        for idx, _ in tqdm(enumerate(fi, 1)):
            pass
    return idx


def main():
    parser = create_parser()
    args = parser.parse_args()
    logger.info(args)

    logger.info('Input file: {}'.format(args.input))
    file_length = count_file_length(args.input)
    logger.info('Number of lines: {}'.format(file_length))

    output_file = path.join(args.output_dir, args.prefix) + ".bpe.{}".format(args.n_vocab)
    logger.info('Training...')
    spm_arg = " ".join(["--model_type=bpe",
                        "--input={input}".format(input=args.input),
                        "--model_prefix={output}".format(output=output_file),
                        "--vocab_size={n_vocab}".format(n_vocab=args.n_vocab),
                        "--character_coverage={coverage}".format(coverage=args.coverage),
                        "--input_sentence_size={sentence_size}".format(sentence_size=file_length),
                        "--control_symbols={}".format(",".join(SPECIAL_TOKENS)),
                        "--shuffle_input_sentence=true",
                        "--add_dummy_prefix=false"])
    logger.info(spm_arg)
    sentencepiece.SentencePieceTrainer.Train(spm_arg)
    logger.info('done')


if __name__ == "__main__":
    main()
