from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable


"""
TODO:
- SegmentTree �����ϱ�
"""


T = TypeVar("T")
U = TypeVar("U")


class SegmentTree(Generic[T, U]):
    #�����ϼ���!
    def __init__(self, size: int):
        """
        ���׸�Ʈ Ʈ�� ������.
        :param size: �迭�� �ִ� ũ�� (���� ���� �ִ� ����).
        """
        self.size = size
        self.tree = [0] * (4 * size)

    def _update(self, node: int, start: int, end: int, idx: int, diff: int):
        """
        ���׸�Ʈ Ʈ���� �����մϴ�. idx ��ġ�� diff ���� ���մϴ�.
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
        idx ��ġ�� diff ���� ���ϴ� ���� �Լ�.
        """
        self._update(1, 1, self.size, idx, diff)

    def _query(self, node: int, start: int, end: int, k: int) -> int:
        """
        ���׸�Ʈ Ʈ������ k��° ��Ҹ� ã���ϴ�.
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
        k��° ��Ҹ� ã�� ���� �Լ�.
        """
        return self._query(1, 1, self.size, k)
    
    def range_sum(self, left: int, right: int) -> int:
        """
        ���׸�Ʈ Ʈ������ [left, right] ������ ���� ���մϴ�.
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
- �ϴ� SegmentTree���� �����ϱ�
- main �����ϱ�
"""

def main() -> None:
    # �����ϼ���!
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
