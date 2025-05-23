#!/usr/bin/env bash
PYTHON_VERSION=$1

set -ex

# Ubuntu 22.04 ships python3.10 by default
apt-get update

if [ -z "$PYTHON_VERSION" ]; then
  DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv git
else
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl software-properties-common git
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get autoremove -y python3
  python_minor_version=$(echo "$PYTHON_VERSION" | awk -F"." '{print $2}')
  if [[ "$python_minor_version" -gt 11 ]]; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-venv"
    rm -f /usr/lib/python*/EXTERNALLY-MANAGED
  else
    DEBIAN_FRONTEND=noninteractive apt-get install -y "python${PYTHON_VERSION}-dev" "python${PYTHON_VERSION}-distutils" "python${PYTHON_VERSION}-venv"
  fi
  ln -sf /usr/bin/"python${PYTHON_VERSION}" /usr/bin/python3
  ln -sf /usr/bin/"python${PYTHON_VERSION}" /usr/bin/python
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
  rm -rf get-pip.py
fi
python3 -m pip --no-cache-dir install -U pip setuptools wheel
python3 -m pip --no-cache-dir install -U "numpy<2" awscli
ln -sf /usr/bin/python3 /usr/bin/python
