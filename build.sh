#!/usr/bin/env bash
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Download precompiled ffmpeg binary
mkdir -p ffmpeg
curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-i686-static.tar.xz -o ffmpeg.tar.xz
tar -xf ffmpeg.tar.xz --strip-components=1 -C ffmpeg
chmod +x ffmpeg/ffmpeg

# Add ffmpeg to PATH so it works in subprocess or os.system
export PATH="$PATH:$PWD/ffmpeg"
