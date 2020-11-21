# ILYS-aoba-chatbot

## 概要

このリポジトリには，「対話システムライブコンペティション3」のオープントラックに提出し，予選を2位で通過した対話システム「ILYS-aoba-bot」の学習済みモデルパラメータ及び学習設定の詳細が記述されたスクリプトが含まれています．


## 配布リソースに関する説明
### ILYS-aoba-bot　系列変換応答生成モジュール
- 本システムでは巨大なパラメータを持つsequence-to-sequence(seq2seq)を系列変換応答生成モジュールとして使用している．seq2seqにはfairseqによって実装されているtransformerを用いている．

### ファイル構成
リリースのリンクと紐づける

|概要|ファイル名|
| ---- | ------------ |
|pretrainedモデル| [ilys\_aoba\_transformer\_pretrained.pt-01](https://github.com/cl-tohoku/ILYS-aoba-chatbot/releases/download/20201104/ilys_aoba_transformer_pretrained.pt-01)|
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
- Python 3.8.
- fairseq==0.9.0
- sentencepiece==0.1.91以上

#### 付属実行コードの利用方法(fine-tunedモデルの場合)
pretrainedモデルを利用する場合はfine-tunedモデルファイルをpretrainedモデルファイルに置き換えて実行してください．

```
# リポジトリのクローン
git clone https://github.com/cl-tohoku/ILYS-aoba-chatbot.git
cd ILYS-aoba-chatbot

# モデル・辞書ファイルをダウンロードしてからファイル辞書ファイルの展開
unzip ilys_aoba_transformer_vocab.zip

# 取得したfine-tunedモデルファイルを１つに統合
cat ilys_aoba_transformer_finetuned.pt-0* > ilys_aoba_transformer_finetuned.pt

# モデルとコマンドラインで対話
python run.py --model ilys_aoba_transformer_finetuned.pt --spm spm_10M_tweets.cr9999.bpe.32000.model --vocab ilys_aoba_transformer_vocab.zip
```


## 引用
引用される方は、以下のbibtexをご利用ください．
予稿集がきたら追記でよさそう

```@article{
hogehoge
}
```


## 連絡先
ご質問等ございましたら、riki.fujihara.s4@dc.tohoku.ac.jpまたはyosuke.kishinami.q8@dc.tohoku.ac.jpへお問い合わせください。

