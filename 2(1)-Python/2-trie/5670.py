from lib import Trie
import sys


"""
TODO:
- 일단 Trie부터 구현하기
- count 구현하기
- main 구현하기
"""


def count(trie: Trie, query_seq: str) -> int:
    """
    trie - 이름 그대로 trie
    query_seq - 단어 ("hello", "goodbye", "structures" 등)

    returns: query_seq의 단어를 입력하기 위해 버튼을 눌러야 하는 횟수
    """
    pointer = 0
    cnt = 0

    for element in query_seq:
        if len(trie[pointer].children) > 1 or trie[pointer].is_end:
            cnt += 1

        new_index = None # 구현하세요!
        for child_index in trie[pointer].children:
            if trie[child_index].body == element:
                new_index = child_index
                break
        else:
            raise ValueError("Character not found in trie, input sequence invalid")

        pointer = new_index

    return cnt + int(len(trie[0].children) == 1)


def main() -> None:
    # 구현하세요!
    """
    표준 입력으로 단어 목록을 읽어 Trie를 구성하고 각 단어를 입력하기 위해 필요한 평균 버튼 입력 횟수를 계산합니다.

    동작:
    1. 입력 데이터를 읽어 테스트 케이스별로 처리합니다.
    2. 각 테스트 케이스에서 Trie를 생성하고 단어를 추가합니다.
    3. 각 단어에 대해 `count`를 호출하여 버튼 입력 횟수를 계산합니다.
    4. 평균 버튼 입력 횟수를 계산하여 소수점 두 자리까지 출력합니다.

    입력 예:
    4
    hello
    hell
    heaven
    goodbye

    출력 예:
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