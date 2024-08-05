# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working dirextory in the containeer
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install system dependencies*
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /vat/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

#Use Poetry to install Python dependencies
RUN /root/.poetry/bin/poetry config virtualenvs.create false \
&& /root/.poetry/bin/poetry install --no-interaction --no-ansi

# Specify the command to run on container start
CMD ["python", "deep_lab/train.py"]