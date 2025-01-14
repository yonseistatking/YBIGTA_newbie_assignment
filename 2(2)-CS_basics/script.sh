#!/bin/bash

# Miniconda 설치 경로
CONDA_DIR="$HOME/miniconda3"

# Miniconda가 존재하지 않을 경우 설치
if [ ! -d "$CONDA_DIR" ]; then
    echo "Miniconda가 설치되지 않았습니다. 설치를 진행합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p "$CONDA_DIR"
    rm ~/miniconda.sh
    export PATH="$CONDA_DIR/bin:$PATH"
    conda init
    echo "Miniconda 설치 완료."
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
source "$CONDA_DIR/bin/activate"
ENV_NAME="myenv"
if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda 가상환경 생성 중: $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.9 -y
fi
conda activate "$ENV_NAME"

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더의 파일 변환 (cp949 -> utf-8)
echo "파일 인코딩 변환을 시작합니다."
for file in submission/*.py; do
    if file "$file" | grep -q "ISO-8859" || file "$file" | grep -q "CP949"; then
        echo "$file 를 UTF-8로 변환 중..."
        iconv -f cp949 -t utf-8 "$file" -o "$file.tmp" && mv "$file.tmp" "$file"
        echo "$file 변환 완료."
    else
        echo "$file 은(는) 변환이 필요하지 않습니다."
    fi
done

# mypy 테스트 실행
echo "mypy 테스트를 시작합니다."
for file in submission/*.py; do
    echo "mypy 테스트: $file"
    mypy "$file"
done

# Submission 폴더의 파일 실행
echo "Python 파일 실행을 시작합니다."
if [ ! -d "output" ]; then
    mkdir -p output
    echo "output 디렉토리를 생성했습니다."
fi

for file in submission/*.py; do
    problem_number=$(basename "$file" .py)  # 파일 이름에서 문제 번호 추출
    input_file="input/${problem_number}_input"
    output_file="output/${problem_number}_output"

    if [[ -f "$input_file" ]]; then
        echo "실행 중: $file (입력 파일: $input_file)"
        python "$file" < "$input_file" > "$output_file"
        echo "출력 저장: $output_file"
    else
        echo "입력 파일 $input_file이 존재하지 않습니다. 스킵합니다."
    fi
done

# 가상환경 비활성화
conda deactivate