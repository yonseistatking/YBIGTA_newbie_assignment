from lib import SegmentTree
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