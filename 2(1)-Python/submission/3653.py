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

    t = int(data[0])  # �׽�Ʈ ���̽��� ��
    results = []
    idx = 1

    for _ in range(t):
        n, m = map(int, data[idx].split())
        movies = list(map(int, data[idx + 1].split()))
        idx += 2

        # ���׸�Ʈ Ʈ�� �ʱ�ȭ
        total_size = n + m
        seg_tree: SegmentTree[int, int] = SegmentTree(total_size)

        # ��ȭ�� �ʱ� ��ġ�� ����
        position = [0] * (n + 1)
        for i in range(1, n + 1):
            position[i] = m + i
            seg_tree.update(position[i], 1)

        # ��ȭ ��û ����
        top = m
        case_result = []

        for movie in movies:
            current_pos = position[movie]
            count_above = seg_tree.range_sum(1, current_pos - 1)
            case_result.append(count_above)

            # ������Ʈ: ���� ��ġ�� �����ϰ�, ���ο� ��ġ(top)�� �߰�
            seg_tree.update(current_pos, -1)
            position[movie] = top
            seg_tree.update(top, 1)
            top -= 1

        results.append(" ".join(map(str, case_result)))

    sys.stdout.write("\n".join(results) + "\n")
    pass


if __name__ == "__main__":
    main()