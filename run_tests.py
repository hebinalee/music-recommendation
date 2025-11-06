#!/usr/bin/env python3
"""
테스트 실행 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests import run_tests


def main():
    """메인 함수"""
    print("🧪 음악 추천 시스템 테스트 시작")
    print("=" * 50)
    
    # 테스트 실행
    success = run_tests()
    
    print("=" * 50)
    if success:
        print("✅ 모든 테스트가 성공적으로 통과했습니다!")
        return 0
    else:
        print("❌ 일부 테스트가 실패했습니다.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

