import sys
import subprocess

print("1. Jalur Python yang sedang aktif:")
print(sys.executable)

print("\n2. Daftar library yang terinstal di jalur ini:")
subprocess.run([sys.executable, "-m", "pip", "list"])