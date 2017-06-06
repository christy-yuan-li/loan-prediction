pip install pandas
pip install sklearn

git clone --recursive https://github.com/Microsoft/LightGBM 
cd LightGBM
mkdir build ; cd build
cmake .. 
make -j 
cd ../python-package; python setup.py install
cd ../../

cd data
git clone https://github.com/petuum-inc/ml-storage.git
mv ml-storage/*.zip ./
unzip train.csv.zip
unzip test_tiny.csv.zip
rm train.csv.zip
rm test_tiny.csv.zip
cd ../
