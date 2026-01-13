#!/bin/bash
# Ralph Once - Single iteration with background timeout

set -e

TIMEOUT=${TIMEOUT:-300}

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Ralph Wiggum Technique - Single Iteration                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check for required files
if [ ! -f "PROMPT.md" ]; then
    echo "❌ Error: PROMPT.md not found"
    echo "   This file contains the main instructions for the AI agent"
    exit 1
fi

if [ ! -f "fix_plan.md" ]; then
    echo "❌ Error: fix_plan.md not found"
    echo "   This file contains the prioritized task list"
    exit 1
fi

# Check if there are any tasks remaining
remaining=$(grep -c '^\- \[ \]' fix_plan.md 2>/dev/null || echo "0")

if [ "$remaining" -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║  ✓ ALL TASKS COMPLETE!                                   ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  - Run tests to verify everything works: (see AGENT.md)"
    echo "  - Create a git tag if all tests pass"
    exit 0
fi

echo "Tasks remaining: $remaining"
echo ""
echo "Reading PROMPT.md and fix_plan.md..."
echo ""

# Run Claude with background timeout watcher
set +e

# Start Claude in background
claude --permission-mode acceptEdits "@PROMPT.md @fix_plan.md @AGENT.md" &
CLAUDE_PID=$!

# Start timeout watcher in background
(
    sleep $TIMEOUT
    if kill -0 $CLAUDE_PID 2>/dev/null; then
        echo ""
        echo "⏱️  TIMEOUT: Killing Claude after ${TIMEOUT}s"
        kill -TERM $CLAUDE_PID 2>/dev/null
        sleep 2
        kill -KILL $CLAUDE_PID 2>/dev/null
    fi
) &
WATCHER_PID=$!

# Wait for Claude to finish
wait $CLAUDE_PID
exit_code=$?

# Kill the watcher if it's still running
kill $WATCHER_PID 2>/dev/null || true

set -e

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Iteration Complete                                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ $exit_code -ne 0 ]; then
    echo "⚠️  Claude exited with code $exit_code"
    echo "   Check the output above for details"
    exit $exit_code
fi

# Check remaining tasks
remaining=$(grep -c '^\- \[ \]' fix_plan.md 2>/dev/null || echo "0")
echo "Tasks remaining: $remaining"

if [ "$remaining" -eq 0 ]; then
    echo ""
    echo "✓ All tasks complete! Final verification needed."
else
    echo ""
    echo "Run './ralph-once.sh' again for the next task."
fi
