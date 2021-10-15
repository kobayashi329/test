# パッケージ管理ツールの更新（apt-getでインストールをするときは必ず行います。）
sudo apt-get -y update
sudo apt-get -y upgrade

# Githubのページを参考にライブラリをダウンロード
# 開発ツール
sudo apt-get -yV install build-essential
sudo apt-get -yV install cmake
# 行列演算
sudo apt-get -yV install libeigen3-dev
# GUIフレームワーク関連
sudo apt-get -yV install libgtk-3-dev
sudo apt-get -yV install qt5-default
sudo apt-get -yV install libvtk7-qt-dev
sudo apt-get -yV install freeglut3-dev
# 並列処理関連
sudo apt-get -yV install libtbb-dev
# 画像フォーマット関連
sudo apt-get -yV install libjpeg-dev
sudo apt-get -yV install libopenjp2-7-dev
sudo apt-get -yV install libpng++-dev
sudo apt-get -yV install libtiff-dev
sudo apt-get -yV install libopenexr-dev
sudo apt-get -yV install libwebp-dev
# 動画像関連
sudo apt-get -yV install libavresample-dev
# その他
sudo apt-get -yV install libhdf5-dev
# Python関連
sudo apt-get -yV install libpython3-dev
sudo apt-get -yV install python3-numpy python3-scipy python3-matplotlib

# gitのインストール（ソースをダウンロードするときに使います。）
sudo apt-get -y install git

# ソースのダウンロード
cd /usr/local
sudo mkdir opencv4
cd /usr/local/opencv4
sudo git clone https://github.com/opencv/opencv.git
sudo git clone https://github.com/opencv/opencv_contrib.git

# ビルド用のディレクトリ作成（buildディレクトリを作成してその中でビルドするのがお作法です。）
cd opencv
sudo mkdir build
cd build

# ビルド
# 基本的にはOpenCV公式ページを参考にしました。
sudo cmake \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=/usr/local/opencv4/opencv_contrib/modules \
PYTHON3_EXECUTABLE=/usr/lib/python3.7 \
PYTHON_INCLUDE_DIR=/usr/include/python3.7 \
PYTHON_INCLUDE_DIR2=/usr/include/arm-linux-gnueabihf/python3.7m \
PYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so \
PYTHON3_NUMPY_INCLUDE_DIRS =/usr/lib/python3/dist-packages/numpy/core/include \
-S /usr/local/opencv4/opencv

sudo make -j7
sudo make install
