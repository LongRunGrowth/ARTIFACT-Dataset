# Download model files in two folders: ./biencoder and ./models
#!/bin/bash

# download modelfiles
wget https://www.dropbox.com/s/tvl4338mv8r06m7/model_files.zip?dl=0

sleep 20
# unzip and clean
unzip 'model_files.zip?dl=0'
rm -rf 'model_files.zip?dl=0'

echo "Model files saved at: ./biencoder and ./models folders"