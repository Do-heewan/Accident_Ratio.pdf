# 1. 파이썬 기반
FROM python:3.8-slim

# 2. 작업 디렉터리 생성
WORKDIR /app

# 3. 프로젝트 파일 복사
COPY requirements.txt requirements.txt
COPY helloworld.py helloworld.py

# 4. 파이썬 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 실행
CMD ["python", "helloworld.py"]