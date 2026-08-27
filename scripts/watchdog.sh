#!/bin/bash
# Watchdog: auto-restart pipeline if it dies
LOG=/root/circuit_ocr/watchdog.log
PIPE=/root/circuit_ocr/scripts/auto_pipeline.py
PYTHON=/root/miniconda3/bin/python
DURATION=${1:-360}

echo "$(date): Watchdog started (duration=${DURATION}min)" >> $LOG

while true; do
    if ! pgrep -f auto_pipeline.py > /dev/null; then
        echo "$(date): Pipeline dead, restarting..." >> $LOG
        cd /root/circuit_ocr
        nohup $PYTHON -u $PIPE --duration $DURATION >> pipeline_console.log 2>&1 &
        echo "$(date): Pipeline PID=$!" >> $LOG
    fi
    sleep 60
done
