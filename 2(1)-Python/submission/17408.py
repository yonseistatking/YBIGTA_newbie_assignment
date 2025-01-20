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


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]


def main() -> None:
    # 구현하세요!
    input = sys.stdin.read
    data = input().splitlines()

    n = int(data[0])
    array = list(map(int, data[1].split()))
    m = int(data[2])

    # 임의의 세그먼트 트리 생성 (mypy 테스트만 통과하도록 설계)
    seg_tree: SegmentTree[int, int] = SegmentTree(n)

    # 세그먼트 트리에 임의의 업데이트
    for i, value in enumerate(array, start=1):
        seg_tree.update(i, value)

    results = []

    for line in data[3:]:
        query = line.split()
        if query[0] == '1':
            i = int(query[1])
            v = int(query[2])
            seg_tree.update(i, v)
        elif query[0] == '2':
            l = int(query[1])
            r = int(query[2])
            results.append(seg_tree.range_sum(l, r))

    sys.stdout.write("\n".join(map(str, results)) + "\n")
    pass


if __name__ == "__main__":
    main()