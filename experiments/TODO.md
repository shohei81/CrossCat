CrossCat GenJAX 実験要件定義書

1. 実験の目的

本実験の主目的は、GenJAXを用いて再実装されたCrossCat（以下、GenJAX版）が、OriginalのPython 2実装（以下、Original版）と比較して、以下の2点を満たしていることを実証することである。

統計的妥当性 (Statistical Validity): Original版と同等の推論精度・挙動を示し、かつ真の生成モデルの構造を復元できること。

計算性能 (Computational Performance): 実用的なシナリオにおいて、Original版よりも高速に収束し、高いスケーラビリティ（行数・列数増加に対する耐性）を有すること。

2. 評価環境と構成

2.1. 比較対象（CPUベースでの比較を主軸とする）

Original版 (Baseline):

環境: Python 2.7 (Docker container)

計算資源: CPUのみ

推論手法: Collapsed Gibbs Sampling

備考: ハイパーパラメータ探索のグリッド数 N_GRID=100。

GenJAX版 CPU (Main Comparison):

環境: Python 3.11+ (JAX)

計算資源: CPU

推論手法: Uncollapsed / Hybrid Gibbs Sampling

目的: アルゴリズム刷新およびJITコンパイルによる純粋な高速化の検証。

GenJAX版 GPU (Additional Experiment):

環境: Python 3.11+ (JAX)

計算資源: NVIDIA GPU

目的: 将来的な大規模データ解析を見据えた、ハードウェアアクセラレーション時の最大性能検証。

2.2. 入出力の共通化

Pythonバージョンの違いを吸収するため、データセットは事前に前処理を行い、共通のCSVファイルとして保存する。

実験結果（ログ）は共通のJSONスキーマで出力し、可視化スクリプトで一元管理する。

3. 使用データセット

Synthetic Data (合成データ)

目的: 真の生成パラメータ（Ground Truth）が既知の状態でのモデルの正当性検証および列数スケーラビリティの厳密な検証。

仕様: パラメータ制御により、列数（次元数）を任意に調整して生成する。

DHA (Dartmouth Health Atlas)

特性: 小規模、連続値中心。

目的: 既存のベンチマークとの回帰テスト。

Adult (Census Income)

特性: 中規模～大規模、数値・カテゴリ混合。

目的: 行数スケーラビリティの検証。サブサンプリングにより行数を段階的に変化させる。

4. 評価指標 (Metrics)

4.1. 統計的妥当性の指標

周辺対数尤度 (Marginal Log-Likelihood: $P(X|Z)$)

定義: 潜在変数 $Z$ に基づき、パラメータ $\theta$ を積分消去した尤度。

実装要件:

GenJAX版においても、パラメータ $\theta$ を持つ状態から一時的に周辺尤度を計算する純粋関数を実装する。

グリッド解像度の統一: Original版の実装に合わせ、ハイパーパラメータ推論時のグリッド分割数を N_GRID=100 に設定する（GenJAXデフォルトの32等から変更する）。

欠損値補完精度 (Imputation Accuracy)

データセットの一部（例: 10%）をランダムに欠損させ、その予測値と真値との誤差（MAE/RMSE）を測定。

依存関係構造の一致度 (Structure Recovery)

Z行列: Original版とGenJAX版の変数間依存確率ヒートマップを比較する。

4.2. 計算性能の指標（スケーラビリティと時間対効果）

収束到達時間 (Time-to-Convergence)

横軸「経過時間（秒）」、縦軸「周辺対数尤度」。

JITコンパイル時間を含めた「実時間」で、Original版と同等の尤度に到達するまでの速さを評価する。

行数・列数スケーラビリティ

データサイズ（$N, D$）に対する1イテレーション平均時間の増加傾向（傾き）を評価。

MCMCステップ数スケーラビリティ

ステップ数増加に伴う JITコンパイル時間の償却（Amortization）効果を確認。

5. 実験手順 (Pipeline)

Step 1: 前処理 (Preprocessing)

各データセットに対し、欠損値マスク処理および正規化を行う。

Adultデータセットのサブセット（100, 1k, 10k, 50k行）を作成。

Step 2: ベンチマーク実行

計測のオーバーヘッドを避けるため、以下の2つのモードに分けて実行する。

Mode A: 速度計測モード (Throughput Measurement)

目的: 純粋な推論速度（iter/sec）とスケーラビリティの計測。

動作:

尤度計算を行わず、MCMC遷移のみを高速に回す。

lax.scan 等でJITコンパイルされたループをノンストップで実行。

計測項目: コンパイル時間、総実行時間、1iterあたりの平均時間。

Mode B: 収束確認モード (Convergence Check)

目的: 時間対精度の検証（Time-to-Accuracy）。

動作:

一定間隔（例: 10 iterごと）または毎ステップ、周辺対数尤度 $P(X|Z)$ を計算する。

計算コストが増加するため、速度計測用としては使用しない。

計測項目: 「経過時間」vs「周辺対数尤度」の推移ログ。

Step 3: ログ収集フォーマット (JSON)

{
  "mode": "speed_check", // or "convergence_check"
  "model": "genjax_cpu",
  "dataset": "adult",
  "rows": 32561,
  "cols": 15,
  "params": { "n_grid": 100, "iterations": 1000 },
  "results": {
    "total_time_sec": 125.4,
    "compilation_time_sec": 5.2,
    "avg_time_per_iter_sec": 0.12,
    "log_likelihood_history": [ ... ] // Mode Bのみ
  }
}


Step 4: 評価・可視化

Scalability Plot (Mode A): 行数・列数に対する実行時間の両対数グラフ。

Convergence Plot (Mode B): 横軸「実時間」・縦軸「尤度」でOriginalとGenJAXを重ねて描画。GenJAXがOriginalの曲線を「左上（短時間で高尤度）」へ追い抜くかを確認。

6. 成功基準 (Success Criteria)

精度: 合成データおよびDHAにおいて、GenJAX版（N_GRID=100）がOriginal版と同等の周辺尤度・構造推定結果に収束すること。

速度: 大規模データ（Adult）において、収束到達時間ベースでGenJAX版（CPU）がOriginal版を上回ること。

拡張性: GenJAX版の実装がGPU実行にも対応し、さらなる高速化の余地（Additional Experiment）を示せること。