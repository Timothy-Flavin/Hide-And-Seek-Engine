#!/bin/bash

echo "Installing project dependencies..."

# Install Pip if necessary, upgrade it
python -m pip install --upgrade pip

# Install required dependencies
pip install torch matplotlib numpy pybind11 gymnasium pettingzoo imageio Pillow

# Install the current package in editable mode
pip install -e .

echo "Installation complete!"
