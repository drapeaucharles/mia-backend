#!/usr/bin/env python3
"""Update the installer with the new miner code"""

# Read the new miner code
with open('mia_miner_heartbeat_v2.py', 'r') as f:
    new_miner_code = f.read()

# Read the installer
with open('install-mia-gpu-miner-v2.sh', 'r') as f:
    installer_content = f.read()

# Find the start and end markers
start_marker = "# Create heartbeat miner with bore and auto-restart\ncat > $INSTALL_DIR/mia_miner_heartbeat.py << 'EOF'\n"
end_marker = "\nEOF\nchmod +x $INSTALL_DIR/mia_miner_heartbeat.py"

# Find positions
start_pos = installer_content.find(start_marker)
if start_pos == -1:
    print("Start marker not found!")
    exit(1)

start_pos += len(start_marker)

end_pos = installer_content.find(end_marker, start_pos)
if end_pos == -1:
    print("End marker not found!")
    exit(1)

# Replace the content
new_installer = installer_content[:start_pos] + new_miner_code + installer_content[end_pos:]

# Write back
with open('install-mia-gpu-miner-v2.sh', 'w') as f:
    f.write(new_installer)

print("Installer updated successfully!")