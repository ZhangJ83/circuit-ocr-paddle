"""Deploy pipeline to cloud and launch."""
import paramiko
import time

HOST = "connect.westc.seetacloud.com"
PORT = 14773
USER = "root"
PASS = "MVehPRdofI3a"

LOCAL_FILES = [
    ("g:/mimo_project/circuit_ocr/scripts/auto_pipeline.py", "/root/circuit_ocr/scripts/auto_pipeline.py"),
]

def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS)
    return c

def run(cmd, desc=""):
    print(f"[{desc}] $ {cmd}")
    return exec_command(cmd)

def exec_command(cmd, timeout=10):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    if out: print(out[-1000:])
    if err: print("STDERR:", err[-500:])
    return out, err

print("=== Connecting ===")
client = connect()
print("Connected!")

# 1. Kill existing processes
print("\n--- Killing stale processes ---")
run("pkill -9 -f train.py 2>/dev/null; pkill -9 -f auto_pipeline.py 2>/dev/null; sleep 2", "Cleanup")
run("nvidia-smi --query-gpu=memory.used --format=csv,noheader", "GPU mem")

# 2. Upload files via SFTP
print("\n--- Uploading files ---")
sftp = client.open_sftp()
for local, remote in LOCAL_FILES:
    print(f"  {local} -> {remote}")
    sftp.put(local, remote)
sftp.close()
print("Upload done!")

# 3. Verify train.py works (quick smoke test)
print("\n--- Smoke test: import check ---")
out, _ = run("cd /root/circuit_ocr && /root/miniconda3/bin/python -c \"import sys; sys.path.insert(0,'scripts'); from train import train; print('IMPORT OK')\" 2>&1", "Import check")

if "IMPORT OK" not in out:
    print("IMPORT FAILED — trying with patches...")
    out, _ = run("cd /root/circuit_ocr && /root/miniconda3/bin/python scripts/test_import2.py 2>&1", "Import check 2")

# 4. Launch pipeline with nohup
print("\n--- Launching pipeline ---")
run("cd /root/circuit_ocr && nohup /root/miniconda3/bin/python -u scripts/auto_pipeline.py > pipeline_console.log 2>&1 &", "Launch")

time.sleep(2)
run("pgrep -af 'auto_pipeline\|train' 2>/dev/null || echo 'no python procs'", "Process check")
run("tail -20 /root/circuit_ocr/pipeline_log.txt 2>/dev/null || echo 'no log yet'", "Pipeline log")

print("\n=== Deploy done ===")
client.close()
