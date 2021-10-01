# ILYS-aoba-chatbot

## 概要

このリポジトリには，「対話システムライブコンペティション3」のオープントラックに提出し，予選を2位で通過した対話システム「ILYS-aoba-bot」の学習済みモデルパラメータ及び学習設定の詳細が記述されたスクリプトが含まれています．


## 配布リソースに関する説明
### ILYS-aoba-bot 系列変換応答生成モジュール
本システムでは巨大なパラメータを持つsequence-to-sequence(seq2seq)を系列変換応答生成モジュールとして使用しています．Seq2seqには[fairseq](https://github.com/pytorch/fairseq)によって実装されているtransformerを用いています．

### ファイル構成

※モデルパラメータは4つのファイルに分割されています．ご利用の際には，4つすべてのファイルをダウンロードする必要があります．

|概要|ファイル名|
| ---- | ------------ |
|pre-trainedモデル| [ilys\_aoba\_transformer\_pretrained.pt-01](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_pretrained.pt-01)|
|| [ilys\_aoba\_transformer\_pretrained.pt-02](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_pretrained.pt-02)|
|| [ilys\_aoba\_transformer\_pretrained.pt-03](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_pretrained.pt-03)|
|| [ilys\_aoba\_transformer\_pretrained.pt-04](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_pretrained.pt-04)|
|fine-tunedモデル| [ilys\_aoba\_transformer\_finetuned.pt-01](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_finetuned.pt-01) |
|| [ilys\_aoba\_transformer\_finetuned.pt-02](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_finetuned.pt-02)|
|| [ilys\_aoba\_transformer\_finetuned.pt-03](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_finetuned.pt-03)|
|| [ilys\_aoba\_transformer\_finetuned.pt-04](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_finetuned.pt-04)|
|sentencepieceモデル|[spm\_10M\_tweets.cr9999.bpe.32000.model](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/spm_10M_tweets.cr9999.bpe.32000.model)
|辞書ファイル|[ilys\_aoba\_transformer\_vocab.zip](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_vocab.zip)|

### 利用方法
#### 実行環境
- Python 3.8以降
- sentencepiece==0.1.91以降
- torch==1.5.1
- torchvision==0.6.1

fairseqについては，このリポジトリのクローン後，以下の手順で導入してください
```
cd ILYS-aoba-chatbot/fairseq
pip install --editable .
```

#### 付属実行コードの利用方法(fine-tunedモデルの場合)
pre-trainedモデルを利用する場合は`finetuned`を`pretrained`に置き換えて実行してください．

```
# リポジトリのクローン
git clone https://github.com/cl-tohoku/ILYS-aoba-chatbot.git
cd ILYS-aoba-chatbot

# モデル・辞書ファイルをダウンロードしてからファイル辞書ファイルの展開
unzip ilys_aoba_transformer_vocab.zip

# 取得したfine-tunedモデルファイルを１つに統合
cat ilys_aoba_transformer_finetuned.pt-0* > ilys_aoba_transformer_finetuned.pt

# モデルとコマンドラインで対話
python run.py --model ilys_aoba_transformer_finetuned.pt --spm spm_10M_tweets.cr9999.bpe.32000.model --vocab fairseq_vocab
```

## ライセンス
The models are distributed under the terms of the [Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

The source code is licensed MIT.


## 引用
引用される方は，以下の文献を引用していただけると幸いです．

```
@inproceedings{ILYS-aoba-bot,
author={藤原吏生 and 岸波洋介 and 今野颯人 and 佐藤志貴 and 佐藤汰亮 and 宮脇峻平 and 加藤拓真 and 鈴木潤 and 乾健太郎},
title={ILYS aoba bot: 大規模ニューラル応答生成モデルとルールベースを統合した雑談対話システム},
booktitle={第90回人工知能学会 言語・音声理解と対話処理研究会(第11回対話システムシンポジウム)},
year={2020}
}
```

## 連絡先
ご質問等ございましたら，riki.fujihara.s4@dc.tohoku.ac.jpまたはyosuke.kishinami.q8@dc.tohoku.ac.jpへお問い合わせください．

