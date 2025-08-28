#!/bin/bash
# Archive old installer scripts

echo "Archiving old installer scripts..."

# Create archive directory
mkdir -p archived_installers

# Move all old installers except universal-installer.sh
for file in *install*.sh *miner*.sh *miner*.py; do
    if [[ "$file" != "universal-installer.sh" && -f "$file" ]]; then
        echo "Archiving: $file"
        mv "$file" archived_installers/
    fi
done

echo "✓ Archived $(ls archived_installers | wc -l) old installer files"
echo "✓ Keeping only: universal-installer.sh"