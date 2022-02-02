#!/bin/bash

# mount additional disk
echo "Sleep... and mount disk... "
sleep 200
# umask 0777
# https://cloud.google.com/compute/docs/disks/add-persistent-disk#formatting
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/sdb
sudo mkdir -p /home/"$GCP_USER"/sdb
sudo mount -o discard,defaults -t ext4 /dev/sdb /home/"$GCP_USER"/sdb

# TODO what of this is truly necessary?
sudo -H -u "$GCP_USER" sudo chmod 777 -R /home/"$GCP_USER"/sdb
sudo -H -u "$GCP_USER" sudo chmod 777 -R /home/"$GCP_USER"/sdb/
sudo -H -u "$GCP_USER" sudo chmod 777 -R /home/"$GCP_USER"/sdb/*
echo "Sleep... and mount disk... finished"
