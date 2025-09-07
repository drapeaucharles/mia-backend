#!/bin/bash
# Check detailed GPU info from backend

echo "🔍 Checking GPU details from backend..."
echo ""

# Get metrics
RESPONSE=$(curl -s https://mia-backend-production.up.railway.app/metrics/gpus)

# Parse with jq if available, otherwise use grep
if command -v jq >/dev/null 2>&1; then
    echo "GPU Details:"
    echo "$RESPONSE" | jq -r '.gpus[0] | "ID: \(.id)\nName: \(.name)\nStatus: \(.status)\nLast heartbeat: \(.seconds_since_heartbeat)s ago"'
    echo ""
    echo "Note: Backend stores IP internally but doesn't expose it in metrics"
else
    echo "$RESPONSE" | grep -o '"id":"[^"]*"' | head -1
    echo "$RESPONSE" | grep -o '"name":"[^"]*"' | head -1
fi

echo ""
echo "💡 To find your public IP, run on your VPS:"
echo "   curl -s https://api.ipify.org"
echo ""
echo "Railway is trying to connect to:"
echo "   http://[YOUR_PUBLIC_IP]:5000/process"
echo ""
echo "To test if port 5000 is accessible:"
echo "   nc -zv [YOUR_PUBLIC_IP] 5000"