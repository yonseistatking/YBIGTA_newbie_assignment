from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    # 구현하세요!
    input = sys.stdin.read
    data = input().splitlines()

    t = int(data[0])  # 테스트 케이스의 수
    results = []
    idx = 1

    for _ in range(t):
        n, m = map(int, data[idx].split())
        movies = list(map(int, data[idx + 1].split()))
        idx += 2

        # 세그먼트 트리 초기화
        total_size = n + m
        seg_tree: SegmentTree[int, int] = SegmentTree(total_size)

        # 영화의 초기 위치를 세팅
        position = [0] * (n + 1)
        for i in range(1, n + 1):
            position[i] = m + i
            seg_tree.update(position[i], 1)

        # 영화 시청 로직
        top = m
        case_result = []

        for movie in movies:
            current_pos = position[movie]
            count_above = seg_tree.range_sum(1, current_pos - 1)
            case_result.append(count_above)

            # 업데이트: 현재 위치를 제거하고, 새로운 위치(top)에 추가
            seg_tree.update(current_pos, -1)
            position[movie] = top
            seg_tree.update(top, 1)
            top -= 1

        results.append(" ".join(map(str, case_result)))

    sys.stdout.write("\n".join(results) + "\n")
    pass


if __name__ == "__main__":
    main()