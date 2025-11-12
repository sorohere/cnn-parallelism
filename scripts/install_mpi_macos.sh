#!/bin/bash

echo "=== MPI Installation Script for macOS ==="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "✓ Homebrew is already installed"
fi

echo ""

# Check if MPI is installed
if command -v mpicc &> /dev/null; then
    echo "✓ MPI is already installed"
    mpicc --version
else
    echo "Installing Open MPI..."
    brew install open-mpi
    
    if [ $? -eq 0 ]; then
        echo "✓ Open MPI installed successfully"
        mpicc --version
    else
        echo "✗ Failed to install Open MPI"
        exit 1
    fi
fi

echo ""
echo "=== Installation Complete ==="
echo ""
echo "You can now compile and run MPI programs:"
echo "  make mnist_data    # Compile data-parallel version"
echo "  make mnist         # Compile pipeline-parallel version"
echo ""

