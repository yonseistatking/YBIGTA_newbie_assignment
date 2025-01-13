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

    n = int(data[0])
    MAX_TASTE = 1_000_000
    seg_tree: SegmentTree = SegmentTree(MAX_TASTE)

    results = []

    for line in data[1:]:
        command = list(map(int, line.split()))
        if command[0] == 1:
            # Find the B-th most delicious candy
            b = command[1]
            result = seg_tree.query(b)
            results.append(result)
            seg_tree.update(result, -1)
        elif command[0] == 2:
            # Add or remove candies of taste B
            b, c = command[1], command[2]
            seg_tree.update(b, c)

    sys.stdout.write("\n".join(map(str, results)) + "\n")
    pass

if __name__ == "__main__":
    main()
