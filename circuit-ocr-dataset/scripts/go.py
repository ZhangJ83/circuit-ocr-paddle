#!/usr/bin/env python3
"""One-click launcher: patches + train. Run inside WSL with: python3 scripts/go.py"""
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = [sys.executable, "scripts/launch_train.py"]
exec(open(sys.argv[1]).read())
