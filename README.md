# 概要
* Kaggle Grandmasterに学ぶ機械学習実践アプローチの勉強記録
* 著者github(https://github.com/abhishekkrthakur/approachingalmost)

## 第1章
* mnistのtSNEを使った可視化

## 第2章
* stratified k-foldの実装

## 第3章
* 様々な評価指標

## 第4章
* 機械学習プロジェクトの実装
  * 決定木，ランダムフォレストでmnistの分類

## 第5章
* データの前処理
  * 欠損値の補完
  * カテゴリ変数のエンコーディング
    * one-hot encoding，label encoding，target encoding，entity embedding
* データセット
  * Categoical Feature Encoding Challenge2，米国の国勢調査データセット
* モデル
  * Scikit-learnのロジスティック回帰，ランダムフォレスト，XGBoost

## 第6章
* 特徴量エンジニアリング
  * datatimeを利用したpandas，ビン化
  * tsfresh
  * scikit-learnのpreprocessingのPolynomialFeatures，KNNImputer(k近傍法による欠損値補完)

## 第7章
* 特徴量選択
  * カイ二乗検定，貪欲法，再帰的特徴量削減(RFE)，モデルベース特徴量選択
* データセット
  * scikit-learnのmake_classification，糖尿病のデータセット
* モデル
  * ロジスティック回帰，ランダムフォレスト

## 第8章
* ハイパーパラメータ最適化
  * グリッドサーチ，ランダムサーチ，パイプライン，ベイズ最適化，hyperopt
* データセット
  * 携帯電話の性能からの価格予測
* モデル
  * ランダムフォレスト

## 第9章
* 画像分類・セグメンテーション
* データセット
  * 肺の気胸の画像データセット(siim-acr-pneumothorax-segmentation)
* モデル
  * ランダムフォレスト，AlexNet，UNet

## 第10章
* 自然言語処理
* データセット
  * IMDBの映画レビューデータセット
* モデル
  * ロジスティック回帰，単純ベイズ分類器
* 前処理
  * CountVectorizer
    * 文章数×単語数の行列を作成
  * TF-IDF
    * 単語の重要度を計算
  * 特異値分解(SVD)
    * 次元削減