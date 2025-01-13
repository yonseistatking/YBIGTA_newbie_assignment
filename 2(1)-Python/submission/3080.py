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
- main �����ϱ�

��Ʈ: �� ����¥�� �ڷῡ�� �׳� str�� ���⿡�� �޸𸮰� �Ʊ���...
"""


def main() -> None:
    # �����ϼ���!


    
    pass


if __name__ == "__main__":
    main()