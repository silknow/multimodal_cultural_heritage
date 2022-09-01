This directory contains the code necessary to run the text classifier.

The experiments were run on Ubuntu Linux with Cuda 11.3 and Python 3.9.
The `install.sh` script creates a python virtual environment using the `requirements.txt` file.

## Run:
./install.sh
source env/bin/activate
./xlmr.py

Some changes have been made to allow for this code to run on GPUs with less memory. Results are not significantly worse or better than in the article. Using a different seed.