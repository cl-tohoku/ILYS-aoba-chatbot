import sentencepiece


class SpmTokenizer:
    def __init__(self, spm_path: str):
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(spm_path)

    def encode(self, text: str) -> str:
        return ' '.join(self.sp.EncodeAsPieces(text))

    def decode(self, text: str) -> str:
        return self.sp.DecodePieces(text.strip().split(' '))
