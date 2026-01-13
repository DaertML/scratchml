#!/bin/bash
# AFK Ralph - Autonomous loop with statistics tracking

# Remove set -e to prevent exit on non-zero codes
# We handle errors explicitly instead

if [ -z "$1" ]; then
    echo "Usage: $0 <iterations>"
    echo ""
    echo "Example: $0 50  # Run 50 iterations"
    echo ""
    echo "Environment variables:"
    echo "  TIMEOUT=300  # Timeout per iteration (default: 300s)"
    exit 1
fi

INSTANCE_ID=${INSTANCE_ID:-"ralph"}
TIMEOUT=${TIMEOUT:-300}

export TIMEOUT  # Make available to ralph-once.sh

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ralph Wiggum Technique - AFK Mode                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo "Instance: $INSTANCE_ID"
echo "Iterations: $1"
echo "Timeout: ${TIMEOUT}s per iteration"
echo ""

# Check for required files
if [ ! -f "PROMPT.md" ]; then
    echo "❌ Error: PROMPT.md not found"
    exit 1
fi

if [ ! -f "fix_plan.md" ]; then
    echo "❌ Error: fix_plan.md not found"
    exit 1
fi

# Statistics
total_successful=0
total_failed=0

for ((i=1; i<=$1; i++)); do
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  Iteration $i/$1 (Instance: $INSTANCE_ID)"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Check if tasks remain
    remaining=$(grep -c '^\- \[ \]' fix_plan.md 2>/dev/null || echo "0")
    
    if [ "$remaining" -eq 0 ]; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║  ✓ ALL TASKS COMPLETE! (Instance: $INSTANCE_ID)           ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "Next steps:"
        echo "  1. Run full test suite (see AGENT.md)"
        echo "  2. Create git tag if all tests pass"
        exit 0
    fi
    
    # Run single iteration - don't let set -e kill us on failure
    set +e
    ./ralph-once.sh
    exit_code=$?
    set -e
    
    # Track statistics
    if [ $exit_code -eq 0 ]; then
        ((total_successful++))
        echo "✓ Iteration $i: Success"
    elif [ $exit_code -eq 2 ]; then
        ((total_failed++))
        echo "⏱️  Iteration $i: Timeout - continuing..."
    else
        ((total_failed++))
        echo "⚠️  Iteration $i: Failed (exit code $exit_code) - continuing..."
    fi
    
    echo "Stats so far: ✓ $total_successful successful | ✗ $total_failed failed"
    
    echo ""
    echo "Pausing 2 seconds before next iteration..."
    sleep 2
done

echo ""
echo "╔════════════════════════════════════════════════════════════╝"
echo "║  Completed $1 iterations (Instance: $INSTANCE_ID)           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Show final status
remaining=$(grep -c '^\- \[ \]' fix_plan.md 2>/dev/null || echo "0")
completed=$(grep -c '^\- \[x\]' fix_plan.md 2>/dev/null || echo "0")

echo "Statistics:"
echo "  Successful iterations: $total_successful"
echo "  Failed iterations: $total_failed"
echo ""
echo "Task Status:"
echo "  Completed tasks: $completed"
echo "  Remaining tasks: $remaining"
echo ""
