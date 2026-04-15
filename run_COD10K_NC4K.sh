#!/bin/bash

echo "===== Start running COD10K ====="
# bash run_COD10K.sh
bash run_COD10K.sh 2>&1 | tee "log$(date +%Y%m%d_%H%M%S)_run_COD10K.txt"

echo "===== Start running NC4K ====="
bash run_NC4K.sh 2>&1 | tee "log$(date +%Y%m%d_%H%M%S)_run_NC4K.txt"

echo "===== All done ====="
