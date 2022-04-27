#!/bin/bash

echo "Sleep... and mount disk... "
sleep 200
source /home/"$GCP_USER"/src/mogen/mogen/Cloud/Startup/basic.sh
echo "Sleep... and mount disk... finished"

python /home/"$GCP_USER"/src/mogen/mogen/Cleaning/redo.py
