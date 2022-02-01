#!/bin/bash

# mount additional disk
echo "Sleep... and mount disk... "
sleep 200
sudo mkfs.ext4 /dev/sdb
sudo mount -t ext4 /dev/sdb /home/"$GCP_USER"/sdb
sudo -H -u "$GCP_USER" sudo chmod 777 -R /home/"$GCP_USER"/sdb
sudo -H -u "$GCP_USER" sudo chmod 777 -R /home/"$GCP_USER"/sdb/
sudo -H -u "$GCP_USER" sudo chmod 777 -R /home/"$GCP_USER"/sdb/*
echo "Sleep... and mount disk... finished"
