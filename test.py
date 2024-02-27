from rich.progress import track
import time

if __name__ == "__main__":
    for i in track(range(100), description="Processing"):
        time.sleep(0.1)