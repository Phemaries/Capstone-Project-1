# First install the python 3.10.13-slim, the slim version uses less space
FROM python:3.10.13-slim

# create a directory in Docker named app and we're using it as work directory
WORKDIR /app

# Install Virtual Environment library in Docker 
RUN python3 -m venv /opt/venv
ENV PATH = "/opt/venv/bin:$PATH"

# Install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# activate virtual environment 
ENV VIRTUAL_ENV = /opt/venv
ENV PATH = "/opt/venv/bin:$PATH"

# Copy the rest of the project files
COPY . .


# Expose the server port
EXPOSE 9696

# Command to start the server
CMD ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict:app"]