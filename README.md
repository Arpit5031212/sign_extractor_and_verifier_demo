##run these commands for quick setup

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install cython
git clone https://github.com/facebookresearch/detectron2
cd detectron2
pip install -e .
pip install opencv-python

pip install paddlepaddle-gpu
pip install paddleocr==2.7.3


## after this first run the signature_extractor.py, then app.py and then use the command streamlit run fe.py to launch the application.
