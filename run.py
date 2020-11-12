import argparse
from os import path

from neural_dialogue_model.model_args import Args
from neural_dialogue_model.models import NeuralDialogueModel


def create_parser():
    parser = argparse.ArgumentParser(description='')
    group = parser.add_argument_group("Dialogues")
    group.add_argument('--model', type=path.abspath, metavar="FP", help="Path to model parameters")
    group.add_argument('--spm', type=path.abspath, metavar="FP", help="Path to sentencepiece model")
    group.add_argument('--vocab', type=path.abspath, metavar="FP", help="Path to vocab")

    return parser


def main():
    parser = create_parser()
    parser_args = parser.parse_args()

    args = Args(model_path=parser_args.model, spm_path=parser_args.spm, vocab_path=parser_args.vocab)

    model = NeuralDialogueModel(args)

    contexts = []
    while True:
        utterance = input("input: ")
        if utterance == "q":
            break
        contexts.append(utterance)
        responses = model(contexts)
        print("output:\n- " + "\n- ".join(responses))
        contexts.append(responses[0])


if __name__ == "__main__":
    main()
