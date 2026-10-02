# MỤC ĐÍCH:  tách ra: giúp chạy từ thư mục gốc dự án với path rõ ràng.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # root

from offline.scraper.main import main

if __name__ == "__main__":
    main()
