from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    #구현하세요!
    def __init__(self, size: int):
        """
        세그먼트 트리 생성자.
        :param size: 배열의 최대 크기 (고유 값의 최대 개수).
        """
        self.size = size
        self.tree = [0] * (4 * size)

    def _update(self, node: int, start: int, end: int, idx: int, diff: int):
        """
        세그먼트 트리를 갱신합니다. idx 위치에 diff 값을 더합니다.
        """
        if idx < start or idx > end:
            return

        self.tree[node] += diff
        if start != end:
            mid = (start + end) // 2
            self._update(node * 2, start, mid, idx, diff)
            self._update(node * 2 + 1, mid + 1, end, idx, diff)

    def update(self, idx: int, diff: int):
        """
        idx 위치에 diff 값을 더하는 공개 함수.
        """
        self._update(1, 1, self.size, idx, diff)

    def _query(self, node: int, start: int, end: int, k: int) -> int:
        """
        세그먼트 트리에서 k번째 요소를 찾습니다.
        """
        if start == end:
            return start

        mid = (start + end) // 2
        if self.tree[node * 2] >= k:
            return self._query(node * 2, start, mid, k)
        else:
            return self._query(node * 2 + 1, mid + 1, end, k - self.tree[node * 2])

    def query(self, k: int) -> int:
        """
        k번째 요소를 찾는 공개 함수.
        """
        return self._query(1, 1, self.size, k)
    
    def range_sum(self, left: int, right: int) -> int:
        """
        세그먼트 트리에서 [left, right] 구간의 합을 구합니다.
        """
        return self._range_sum(1, 1, self.size, left, right)

    def _range_sum(self, node: int, start: int, end: int, left: int, right: int) -> int:
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return self._range_sum(node * 2, start, mid, left, right) + self._range_sum(node * 2 + 1, mid + 1, end, left, right)
    
    
    
    pass


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