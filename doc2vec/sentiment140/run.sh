#!/bin/bash

python sentiment140_clean.py
python build_model.py

echo "Now you need to run process.py and provide options"
