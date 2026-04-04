#!/bin/bash

echo "Installing project dependencies..."

# Install Pip if necessary, upgrade it
python -m pip install --upgrade pip

# Install required dependencies
<<<<<<< HEAD
pip install torch matplotlib numpy pybind11 gymnasium pettingzoo imageio Pillow
=======
pip install torch matplotlib numpy pybind11 gymnasium pettingzoo imageio Pillow minigrid
>>>>>>> optimbranch

# Install the current package in editable mode
pip install -e .

echo "Installation complete!"
