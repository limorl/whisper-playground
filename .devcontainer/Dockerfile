
# Default image for DevContainer: Python 3.10 for Apple Silicone
ARG IMAGE="mcr.microsoft.com/vscode/devcontainers/python"
ARG VERSION="3.11"
ARG VARIANT="-bullseye"

FROM ${IMAGE}:${VERSION}${VARIANT}

# Default target architecture. For another architecture, change .env file
ARG TARGETARCH="arm64"

# Install Docker CE CLI
RUN apt-get update \
    && apt-get install -y apt-transport-https ca-certificates curl gnupg2 lsb-release \
    && curl -fsSL https://download.docker.com/linux/$(lsb_release -is | tr '[:upper:]' '[:lower:]')/gpg | apt-key add - 2>/dev/null \
    && echo "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/$(lsb_release -is | tr '[:upper:]' '[:lower:]') $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli

# Install Docker Compose
RUN export LATEST_COMPOSE_VERSION=$(curl -sSL "https://api.github.com/repos/docker/compose/releases/latest" | grep -o -P '(?<="tag_name": ").+(?=")') \
    && curl -sSL "https://github.com/docker/compose/releases/download/${LATEST_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Update and install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common zip unzip && \
    apt-get clean

# making git case sensetive, so rename from File.js to file.js will be makred as a change
RUN git config --global core.ignorecase false

# Pip cach-dir
# pRUN pip --cache-dir "${WORKSPACE_DIR}/.cache"

# Install whisper
RUN pip install git+https://github.com/openai/whisper.git


