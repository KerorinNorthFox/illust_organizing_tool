# illust_organizing_tools

イラストの整理に関するスクリプト群

## 重複画像の検出

pytorchのResNet50とコサイン類似度を用いて重複画像を判定する

### `detect_same_image.py`

判定を行うコードを実装

### `isolate_duplicated_images.py`

`detect_same_image.py`を用いて、実際に同一ディレクトリ内の重複画像を別ディレクトリに移動するスクリプト

計算量がO\(n^2\)となっており非常に遅いため、FAISSを用いたベクトルベースの計算手法に変更予定。
