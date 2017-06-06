pip install pandas 
pip install sklearn
git clone --recursive https://github.com/Microsoft/LightGBM 
cd LightGBM
mkdir build ; cd build
cmake .. 
make -j 
cd ../python-package; python setup.py install
cd ../../
