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
