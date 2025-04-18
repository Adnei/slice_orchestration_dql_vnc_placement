#!/bin/bash

python -m venv .venv
source .venv/bin/activate
.venv/bin/pip install -r requirements.txt
.venv/bin/pip --upgrade pip
.venv/bin/pip install --upgrade pip
