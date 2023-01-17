# Download data files
#!/bin/bash

GREEN='\033[0;32m'
CYAN='\033[0;36m'  
NC='\033[0m'
# download data files
mkdir -p ../data
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=16-1lB_8Sbq30lAHIOVMYOBLeGGs2O5O8' -O ../data/CHOD-Dataset.zip
clear
printf "${GREEN}Data downloaded at ../data/CHOD-Dataset.zip${NC}\n"
sleep 2

# unzip and clean
# printf "${GREEN}Uncompressing ../data/CHOD-Dataset.zip${NC}\n"
# unzip '../data/CHOD-Dataset.zip' -d '../data/CHOD-Dataset'
# sleep 1
# rm -rf '../data/CHOD-Dataset.zip'
