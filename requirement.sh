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
unzip train_v2.csv.zip
unzip test_v2.csv.zip
rm train_v2.csv.zip
rm test_v2.csv.zip
cd ../
