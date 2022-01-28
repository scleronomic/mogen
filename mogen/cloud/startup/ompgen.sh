#!/bin/bash
touch /home/"$GCP_USER"/startup_test0.txt
source /home/"$GCP_USER"/src/mogen/mogen/cloud/startup/basic.sh

nohup python /home/"$GCP_USER"/src/mogen/mogen/generation/path.py
touch /home/"$GCP_USER"/startup_test1.txt
