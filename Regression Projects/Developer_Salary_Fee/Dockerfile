FROM python:3.11-slim

WORKDIR /app

# Install system deps (optional but recommended)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy notebook and data
COPY stackoverflow-2022-salary-prediction.ipynb .
COPY ./data/survey_results_public.csv .

# Run notebook and export output
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
