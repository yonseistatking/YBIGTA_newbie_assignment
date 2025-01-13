from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Iterable

"""
TODO:
- Trie.push �����ϱ�
- (�ʿ��� ���) Trie�� �߰� method �����ϱ�
"""

T = TypeVar("T")

@dataclass
class TrieNode(Generic[T]):
    body: Optional[T] = None
    children: list[int] = field(default_factory=lambda: [])
    is_end: bool = False

class Trie(list[TrieNode[T]]):
    def __init__(self) -> None:
        super().__init__()
        self.append(TrieNode(body=None))

    def push(self, seq: Iterable[T]) -> None:
        """
        seq: T�� �� (list[int]�� ���� �ְ� str�� ���� �ְ� ���...)

        action: trie�� seq�� �����ϱ�
        """
        # �����ϼ���!
        current_index = 0
        for element in seq:
            for child_index in self[current_index].children:
                if self[child_index].body == element:
                    current_index = child_index
                    break
            else:
                new_index = len(self)
                self.append(TrieNode(body=element))
                self[current_index].children.append(new_index)
                current_index = new_index
        self[current_index].is_end = True
        

    
    




import sys


"""
TODO:
- �ϴ� Trie���� �����ϱ�
- count �����ϱ�
- main �����ϱ�
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - �̸� �״�� trie
    query_seq - �ܾ� ("hello", "goodbye", "structures" ��)

    returns: query_seq�� �ܾ �Է��ϱ� ���� ��ư�� ������ �ϴ� Ƚ��
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = None # �����ϼ���!
        for child_index in trie[pointer].children:
            if trie[child_index].body == element:
                new_index = child_index
                break
        else:
            raise ValueError("Character not found in trie, input sequence invalid")

        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    # �����ϼ���!
    """
    ǥ�� �Է����� �ܾ� ����� �о� Trie�� �����ϰ� �� �ܾ �Է��ϱ� ���� �ʿ��� ��� ��ư �Է� Ƚ���� ����մϴ�.

    ����:
    1. �Է� �����͸� �о� �׽�Ʈ ���̽����� ó���մϴ�.
    2. �� �׽�Ʈ ���̽����� Trie�� �����ϰ� �ܾ �߰��մϴ�.
    3. �� �ܾ ���� `count`�� ȣ���Ͽ� ��ư �Է� Ƚ���� ����մϴ�.
    4. ��� ��ư �Է� Ƚ���� ����Ͽ� �Ҽ��� �� �ڸ����� ����մϴ�.

    �Է� ��:
    4
    hello
    hell
    heaven
    goodbye

    ��� ��:
    2.00
    """
    input = sys.stdin.read
    data = input().splitlines()

    index = 0
    results = []

    while index < len(data):
        N = int(data[index])
        index += 1

        words = data[index:index + N]
        index += N

        trie: Trie = Trie()
        for word in words:
            trie.push(word)

        total_keystrokes = sum(count(trie, word) for word in words)

        avg_keystrokes = total_keystrokes / N
        results.append(f"{avg_keystrokes:.2f}")

    print("\n".join(results))
    pass


if __name__ == "__main__":
    main()