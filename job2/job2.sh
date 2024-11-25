#!/bin/sh

### -------------specify queue name ----------------
#BSUB -q c02516

### -------------specify GPU request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### -------------specify job name ----------------
#BSUB -J yolo_model_with_log

### -------------specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### -------------specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"

### -------------specify wall-clock time (max allowed is 12:00)----------------
#BSUB -W 12:00
#BSUB -o OUTPUT_FILE%J.out
#BSUB -e OUTPUT_FILE%J.err

# Cleanup previous files (optional)
rm -f /zhome/c7/f/213256/pfas/job2/*.pt

# Activate the virtual environment
source ~/myenvs/myenv/bin/activate
if [ $? -ne 0 ]; then
  echo "Virtual environment activation failed!"
  exit 1
fi

# Run the training script
python3 /zhome/c7/f/213256/pfas/job2/yolo_model_with_log.py

# Check the exit status
if [ $? -eq 0 ]; then
  echo "Job completed successfully!"
else
  echo "Job encountered an error!"
  exit 1
fi












