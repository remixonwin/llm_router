#!/usr/bin/env python3
import os
import subprocess
import signal
import sys
import argparse

def kill_port(port: int):
    """Identify and terminate the process using the specified port."""
    print(f"Checking for processes on port {port}...")
    
    # Try using lsof
    try:
        output = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        if output:
            pids = output.split("\n")
            for pid in pids:
                pid = int(pid)
                print(f"Terminating process {pid} using port {port}...")
                os.kill(pid, signal.SIGTERM)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback to netstat/fuser if needed (Linux specific)
    try:
        output = subprocess.check_output(["fuser", f"{port}/tcp"], stderr=subprocess.STDOUT, text=True).strip()
        if output:
            print(f"fuser found process on {port}: {output}")
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    print(f"No active process found on port {port}.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kill process on a specific port.")
    parser.add_argument("port", type=int, help="Port number to clear")
    args = parser.parse_args()
    
    kill_port(args.port)
