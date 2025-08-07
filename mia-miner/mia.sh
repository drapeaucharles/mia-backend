#!/bin/bash

# MIA Miner One-Line Installer
# Usage: curl -sSL https://yourdomain.com/mia.sh | bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "███╗   ███╗██╗ █████╗     ███╗   ███╗██╗███╗   ██╗███████╗██████╗ "
echo "████╗ ████║██║██╔══██╗    ████╗ ████║██║████╗  ██║██╔════╝██╔══██╗"
echo "██╔████╔██║██║███████║    ██╔████╔██║██║██╔██╗ ██║█████╗  ██████╔╝"
echo "██║╚██╔╝██║██║██╔══██║    ██║╚██╔╝██║██║██║╚██╗██║██╔══╝  ██╔══██╗"
echo "██║ ╚═╝ ██║██║██║  ██║    ██║ ╚═╝ ██║██║██║ ╚████║███████╗██║  ██║"
echo "╚═╝     ╚═╝╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo "GPU Mining Client Installer"
echo "=================================="
echo ""

# Check system
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "This installer is for Linux systems only"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip git curl wget > /dev/null 2>&1

# Set up environment
INSTALL_DIR="$HOME/mia-miner"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Clone repository
if [ ! -d ".git" ]; then
    echo "Downloading MIA miner..."
    git clone -q https://github.com/drapeaucharles/mia-backend.git temp_clone
    mv temp_clone/mia-miner/* .
    rm -rf temp_clone
fi

# Install Python dependencies
echo "Installing Python packages..."
pip3 install -q requests

# Install Golem silently in background
echo "Setting up compute backend..."
export YAGNA_PAYMENT_RECEIVER="0x690E879Bbb1738827b0891Bbe00F6863AC91BA76"

# Download and install yagna in background
{
    mkdir -p "$HOME/.local/bin"
    cd "$HOME/.local/bin"
    
    # Download yagna
    if [ ! -f "yagna" ]; then
        curl -sSL https://github.com/golemfactory/yagna/releases/download/v0.13.2/golem-provider-linux-v0.13.2.tar.gz | tar -xz --strip-components=1
    fi
    
    # Download golemsp
    if [ ! -f "golemsp" ]; then
        curl -sSL https://github.com/golemfactory/ya-provider/releases/download/v0.13.0/golemsp-linux-v0.13.0.tar.gz | tar -xz
    fi
    
    # Add to PATH
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
} > /dev/null 2>&1 &

# Get configuration
cd "$INSTALL_DIR"
echo ""
echo "Configuration"
echo "-------------"

# Default values
DEFAULT_API_URL="https://mia-backend-production.up.railway.app"
DEFAULT_MINER_NAME="mia-gpu-$(hostname)-$(date +%s)"

# Use environment variables if set, otherwise prompt
if [ -z "$MIA_API_URL" ]; then
    read -p "MIA API URL [$DEFAULT_API_URL]: " MIA_API_URL
    MIA_API_URL=${MIA_API_URL:-$DEFAULT_API_URL}
fi

if [ -z "$MINER_NAME" ]; then
    read -p "Miner Name [$DEFAULT_MINER_NAME]: " MINER_NAME
    MINER_NAME=${MINER_NAME:-$DEFAULT_MINER_NAME}
fi

# Create environment file
cat > "$INSTALL_DIR/.env" << EOF
MIA_API_URL=$MIA_API_URL
MINER_NAME=$MINER_NAME
POLL_INTERVAL=5
YAGNA_PAYMENT_RECEIVER=0x690E879Bbb1738827b0891Bbe00F6863AC91BA76
PATH=$HOME/.local/bin:\$PATH
EOF

# Create start script
cat > "$INSTALL_DIR/start_miner.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source .env
export PATH="$HOME/.local/bin:$PATH"
python3 run_miner.py
EOF
chmod +x "$INSTALL_DIR/start_miner.sh"

# Create systemd service (optional)
if command -v systemctl &> /dev/null; then
    echo ""
    read -p "Install as system service? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SERVICE_FILE="/tmp/mia-miner.service"
        cat > "$SERVICE_FILE" << EOF
[Unit]
Description=MIA GPU Miner
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$HOME/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$INSTALL_DIR/start_miner.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        sudo mv "$SERVICE_FILE" /etc/systemd/system/mia-miner.service
        sudo systemctl daemon-reload
        sudo systemctl enable mia-miner.service
        sudo systemctl start mia-miner.service
        
        echo -e "${GREEN}✓ MIA miner installed as system service${NC}"
        echo ""
        echo "Service commands:"
        echo "  Check status: sudo systemctl status mia-miner"
        echo "  View logs:    sudo journalctl -u mia-miner -f"
        echo "  Stop:         sudo systemctl stop mia-miner"
        echo "  Start:        sudo systemctl start mia-miner"
    else
        # Start in background
        nohup "$INSTALL_DIR/start_miner.sh" > "$INSTALL_DIR/miner.log" 2>&1 &
        MINER_PID=$!
        echo $MINER_PID > "$INSTALL_DIR/miner.pid"
        
        echo -e "${GREEN}✓ MIA miner started (PID: $MINER_PID)${NC}"
        echo ""
        echo "Commands:"
        echo "  View logs:    tail -f $INSTALL_DIR/miner.log"
        echo "  Stop:         kill \$(cat $INSTALL_DIR/miner.pid)"
        echo "  Start:        $INSTALL_DIR/start_miner.sh"
    fi
else
    # No systemd, start in background
    nohup "$INSTALL_DIR/start_miner.sh" > "$INSTALL_DIR/miner.log" 2>&1 &
    MINER_PID=$!
    echo $MINER_PID > "$INSTALL_DIR/miner.pid"
    
    echo -e "${GREEN}✓ MIA miner started (PID: $MINER_PID)${NC}"
    echo ""
    echo "Commands:"
    echo "  View logs:    tail -f $INSTALL_DIR/miner.log"
    echo "  Stop:         kill \$(cat $INSTALL_DIR/miner.pid)"
    echo "  Start:        $INSTALL_DIR/start_miner.sh"
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo "Miner location: $INSTALL_DIR"
echo ""