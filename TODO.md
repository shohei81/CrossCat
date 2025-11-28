論文執筆に向けた厳密な比較検証、素晴らしいですね。既存のGenJAX実装が持つ「有限切断（Finite Approximation）されたSBP（Stick-Breaking Process）」という特性と、比較対象である `probcomp/crosscat`（通常は無限混合モデルとして実装されるCRP）の違いを意識しつつ、公平なベンチマークを行うためのTODOを再構成しました。

### 1. 技術選定と環境構築（Environment Isolation）
依存関係の競合（特に `probcomp/crosscat` は古いC++コンパイラやPython 2.7/3.5あたりを要求する可能性があります）を完全に排除するため、**Docker** の利用を強く推奨します。

* **Dockerの採用:**
    * **Legacy Container (`probcomp`用):** 古いOS（例: Ubuntu 16.04/18.04）ベースで環境構築し、`crosscat` サーバーを立ち上げるか、CLIで実行できる状態にします。
    * **Modern Container (`GenJAX`用):** 最新のCUDA/JAX環境（`uv` で管理されている現在の環境）をコンテナ化します。
* **データ共有:** ホスト側のディレクトリを両方のコンテナにマウントし、同一のCSVデータと設定ファイルを読み込ませます。

### 2. 統一すべきハイパーパラメータ（Hyperparameter Alignment）
論文の公平性を担保するため、以下のパラメータを数理的な意味で一致させる必要があります。GenJAX側は `src/crosscat/constants.py` で定義されていますが、これを `probcomp/crosscat` のJSON設定（metadata/schema）と合わせます。

| パラメータの意味 | GenJAX定数名 | probcomp/crosscat 対応項目 (推定) | 補足・注意点 |
| :--- | :--- | :--- | :--- |
| **ビュー（View）の集中度** | `ALPHA_VIEW` | `alpha` (View CRP hyperparam) | Viewの生成されやすさを制御します。 |
| **クラスタの集中度** | `ALPHA_CLUSTER` | `alpha` (Cluster CRP hyperparam) | 各View内でのクラスタの生成されやすさです。 |
| **カテゴリカルデータの集中度** | `ALPHA_CAT` | Dirichlet prior (`alpha`) | カテゴリ分布の事前分布です。 |
| **数値データの事前平均** | `MU0_PRIOR_MEAN` | `mu` / `m` (Normal-Gamma hyper) | データの中心位置の事前期待値です。 |
| **数値データの事前分散** | `MU0_PRIOR_VAR` | `k` / `r` scaling factor | 平均の確信度に関わります。 |
| **Normal-Gammaパラメータ** | `NG_ALPHA0`, `NG_BETA0`, `NG_KAPPA0` | `nu`, `s`, `beta` など | 分散の事前分布（逆ガンマ分布等）のパラメータです。数式を確認し、$\alpha, \beta$ の定義が一致しているか確認が必要です。 |
| **切断数（Truncation）** | `NUM_VIEWS`, `NUM_CLUSTERS` | **設定なし（無限）** | **【最重要】** GenJAXは有限近似（固定長）ですが、本家は無限モデルです。論文では「十分大きな数（例: データ数と同等かそれ以上）を設定し、実質的に近似誤差を無視できる状態」にするか、比較対象側にも上限を設ける設定があるか確認が必要です。 |

### 3. TODOリスト

#### Phase 1: 環境とデータの準備
1. **Docker環境の構築**
    * `probcomp/crosscat` が動作するDockerイメージを作成する（C++バックエンドのビルドを含む）。
    * GenJAX用のDockerイメージを作成する（GPU利用可能にする）。
        * `docker/legacy/Dockerfile`, `docker/genjax/Dockerfile`, および `docker/docker-compose.yml` でベース環境を定義済み。READMEの「Dockerized environments」を参照し、必要なマウントや追加パッケージを状況に合わせて拡張する。
2. **実データの前処理パイプライン作成**
    * ターゲットとする実データCSVを取得する。
    * 以下の2つの形式に変換するスクリプトを作成する。
        * **GenJAX用:** 数値データ行列 (`rows_cont`) とカテゴリデータ行列 (`rows_cat`) の `numpy` 配列 。
        * **CrossCat用:** CSVファイルおよび、列の型定義（Numerical/Categorical）を含むJSONスキーマ。
    * データサイズ（行数 $N$、列数 $D$）を変えたサブセット（例: 10%, 50%, 100%）を作成し、スケーラビリティ検証の準備をする。

#### Phase 2: 条件の厳密な統一
3. **ハイパーパラメータの固定**
    * 上記の表に基づき、共通の設定ファイル（YAMLやJSONなど）を作成する。
    * GenJAX側: 設定ファイルを読み込み、`constants.py` の値を動的に上書きする仕組みを実装する。
    * CrossCat側: 設定ファイルを読み込み、エンジンの初期化パラメータに変換するスクリプトを実装する。
4. **モデル構造の整合性確認**
    * GenJAX側の `NUM_VIEWS`, `NUM_CLUSTERS` を、実データに対して十分大きな値（サンプリング中に上限に達しない値）に設定し、ノンパラメトリック的な挙動を模倣させる。

#### Phase 3: 計測と実行
5. **ウォームアップと計測基準の統一**
    * **GenJAX:** JITコンパイル時間を含めない「純粋なサンプリング時間（Iter/sec）」と、コンパイル込みの「Total時間」を分けて計測する実装にする（既存テストコード の `test_timing_gibbs_sweep_jit` が参考になります）。
    * **CrossCat:** データのロード時間を除いた、遷移（Transition）にかかる時間を計測する。
6. **ベンチマークの実行**
    * 同一のイテレーション回数（例: 100回、1000回）で両者を実行。
    * 各イテレーションごとの対数尤度（Log Likelihood）を記録し、**収束速度（時間あたりの尤度の改善率）** も比較データとして取得する（単なるループ速度だけでなく、推論の質も比較するため）。

#### Phase 4: 論文用データの整理
7. **結果の可視化**
    * 横軸：時間（秒）、縦軸：対数尤度のグラフを作成（収束性能の比較）。
    * 横軸：データサイズ、縦軸：1イテレーションあたりの時間のグラフを作成（スケーラビリティの比較）。
8. **ボトルネック分析（Discussion用）**
    * GenJAXが速い理由（`vmap`, `scan` による並列化）と、遅い場合（コンパイルオーバーヘッド、メモリ制約）の考察を行うためのプロファイリングデータを取得する。
