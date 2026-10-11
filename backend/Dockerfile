FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PORT=8080 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD exec gunicorn --bind :${PORT} --worker-class sync --workers 1 --threads 1 --timeout 120 main:app
