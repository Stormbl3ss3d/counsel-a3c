#!/bin/bash

cd /counsel

echo "======================================"
echo "Starting A3C Parallel Training (8 Models)..."
echo "======================================"

# Record the start time
START_TIME=$(date +%s)

# Run the 8 training configurations in sequence with 4 parallel workers
python3 traina3c.py -n std-f5-c3 -ncf 5 -ncp 3 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f5-c5 -ncf 5 -ncp 5 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f5-c10 -ncf 5 -ncp 10 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f5-c20 -ncf 5 -ncp 20 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f10-c3 -ncf 10 -ncp 3 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f25-c3 -ncf 25 -ncp 3 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f50-c3 -ncf 50 -ncp 3 -nw 4 -e 1000 &&
python3 traina3c.py -n std-f100-c3 -ncf 100 -ncp 3 -nw 4 -e 1000

# Record the end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Convert seconds to minutes and seconds
MINS=$((DURATION / 60))
SECS=$((DURATION % 60))

# Output to the console
echo "======================================"
echo "Training Complete!"
echo "Total Time: $MINS minutes and $SECS seconds."
echo "======================================"

# Save the time to a text file on your laptop for later comparison
echo "A3C 8-Model Run (-nw 4) Total Training Time: $MINS minutes and $SECS seconds." > logs/parallel_training_time_log.txt
