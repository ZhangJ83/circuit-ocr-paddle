#!/bin/bash
# Clean C: drive temp files to free space
echo "=== Disk before cleanup ===" > /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
df -h /mnt/c >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt 2>&1

echo "=== Cleaning claude temp ===" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
rm -rf /mnt/c/Users/zzz/AppData/Local/Temp/claude 2>/dev/null
echo "exit: $?" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt

echo "=== Cleaning .claude projects ===" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
rm -rf /mnt/c/Users/zzz/.claude/projects 2>/dev/null
echo "exit: $?" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt

echo "=== Cleaning pip cache ===" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
rm -rf /mnt/c/Users/zzz/AppData/Local/pip/cache 2>/dev/null
echo "exit: $?" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt

echo "=== Cleaning Windows Temp ===" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
find /mnt/c/Users/zzz/AppData/Local/Temp -maxdepth 1 -name '*.tmp' -delete 2>/dev/null
find /mnt/c/Users/zzz/AppData/Local/Temp -maxdepth 1 -name 'tmp*' -type d -exec rm -rf {} + 2>/dev/null
echo "exit: $?" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt

echo "=== Disk after cleanup ===" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
df -h /mnt/c >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt 2>&1

echo "DONE" >> /mnt/g/mimo_project/circuit_ocr/cleanup_log.txt
