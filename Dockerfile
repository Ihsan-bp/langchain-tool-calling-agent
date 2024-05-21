# Use an official Python image as the base image for local development
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install OS dependencies required for Python packages
RUN apt-get update && \
    apt-get install -y \
    gcc \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev && \
    apt-get clean

# Copy the requirements.txt file into the container at /var/task
COPY ./Agent_4all/requirements.txt .

# Install Python dependencies using pip
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application into the container
COPY ./google-sheet .

ENV OPENAI_API_KEY=sk-proj-LoaF6RldARSrzMP3fgbfT3BlbkFJFXic1rzuNTEme5VOvmOx
ENV MONGODB_CONNECTION_STRING=mongodb+srv://niteshsinghal9917:bzDCS9Hmf6EOvXxP@cluster0.si0lxp7.mongodb.net/

# # Set environment variables
# ENV AWS_LAMBDA_FUNCTION_HANDLER=lambda_function.handle_request("test_event",{})
# # Set MPLCONFIGDIR to a writable directory within /tmp
# ENV MPLCONFIGDIR=/tmp/matplotlib
# ENV HF_HOME=/tmp/huggingface

# Set the CMD to your handler function. This will execute when the container starts.
CMD ["python3", "-c", "from lambda_function import handle_request; handle_request("test_event",{})"]