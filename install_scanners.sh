#!/bin/bash
# Helper script to install Checkov and tfsec for Axon InfraLinter

set -e

echo "=========================================="
echo "Installing Security Scanners"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  WARNING: Virtual environment doesn't appear to be activated"
    echo "   Please activate your venv first:"
    echo "   source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Checkov
echo "1. Installing Checkov..."
pip install checkov

# Verify Checkov installation
echo ""
echo "Verifying Checkov installation..."
if python -m checkov --version > /dev/null 2>&1; then
    echo "✓ Checkov installed successfully"
    python -m checkov --version
else
    echo "✗ Checkov installation verification failed"
    echo "  Try: pip install --upgrade checkov"
fi

# Install tfsec
echo ""
echo "2. Installing tfsec..."
if command -v tfsec &> /dev/null; then
    echo "✓ tfsec is already installed"
    tfsec --version
else
    echo "Downloading tfsec..."
    TFSEC_VERSION=$(curl -s https://api.github.com/repos/aquasecurity/tfsec/releases/latest | grep tag_name | cut -d '"' -f 4)
    echo "Latest version: $TFSEC_VERSION"
    
    wget -q "https://github.com/aquasecurity/tfsec/releases/download/${TFSEC_VERSION}/tfsec-linux-amd64" -O /tmp/tfsec
    chmod +x /tmp/tfsec
    
    # Try to install to user-local bin first (no sudo needed)
    if [ -d "$HOME/.local/bin" ]; then
        mkdir -p "$HOME/.local/bin"
        mv /tmp/tfsec "$HOME/.local/bin/tfsec"
        echo "✓ tfsec installed to ~/.local/bin/tfsec"
        echo "  Make sure ~/.local/bin is in your PATH"
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Installing tfsec to /usr/local/bin (requires sudo)..."
        sudo mv /tmp/tfsec /usr/local/bin/tfsec
        echo "✓ tfsec installed to /usr/local/bin/tfsec"
    fi
fi

# Verify tfsec installation
echo ""
echo "Verifying tfsec installation..."
if command -v tfsec &> /dev/null; then
    echo "✓ tfsec installed successfully"
    tfsec --version
else
    echo "✗ tfsec not found in PATH"
    echo "  If installed to ~/.local/bin, add to PATH:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the diagnostic: python -m axon_infralinter.scanning.diagnose_scans"
echo "2. If both tools are detected, run the scanner: python -m axon_infralinter.scanning.scanner"


