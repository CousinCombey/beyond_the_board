FROM python:3.12.9
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY beyond_the_board /beyond_the_board
CMD uvicorn beyond_the_board.api.chess:app --host 0.0.0.0 --port $PORT
