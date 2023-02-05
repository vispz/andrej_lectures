scp ~/work/ml_learning/andrej_lectures/input.txt ubuntu@129.213.90.199:~/andrej_lectures/
git clone https://github.com/psvishnu91/andrej_lectures.git
cp input.txt andrej_lectures/
virtualenv ~/venv3
source ~/venv3/bin/activate
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu117
pip3 install -U tensorboard tqdm
