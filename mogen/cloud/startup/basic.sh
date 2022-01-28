#!/bin/bash
sleep 300
sudo mkfs.ext4 /dev/sdb
sudo mount -t ext4 /dev/sdb /home/"$GCP_USER"/sdb

python /home/"$GCP_USER"/src/wzk/wzk.git2.py

touch /home/"$GCP_USER"/startup_test1a.txt