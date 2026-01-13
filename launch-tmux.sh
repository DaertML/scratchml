#!/bin/bash
# Save as: launch-tmux.sh

SESSION_NAME="claude-ralph"

tmux kill-session -t $SESSION_NAME 2>/dev/null

echo "Starting Ralph instances..."
docker-compose up -d
sleep 3

echo "Creating tmux session..."
tmux new-session -d -s $SESSION_NAME -n "Ralph"
tmux split-window -h -t $SESSION_NAME:0
tmux split-window -v -t $SESSION_NAME:0.0
tmux split-window -v -t $SESSION_NAME:0.2

echo "Connecting to containers..."
tmux send-keys -t $SESSION_NAME:0.0 "docker exec -it claude-ai-1 bash" C-m
tmux send-keys -t $SESSION_NAME:0.1 "docker exec -it claude-ai-2 bash" C-m
tmux send-keys -t $SESSION_NAME:0.2 "docker exec -it claude-ai-3 bash" C-m
tmux send-keys -t $SESSION_NAME:0.3 "docker exec -it claude-ai-4 bash" C-m

sleep 2

echo "Configuring Claude (one-time setup)..."

# Start claude with a simple prompt to trigger setup dialogs
tmux send-keys -t $SESSION_NAME:0.0 "claude 'hello'" C-m
tmux send-keys -t $SESSION_NAME:0.1 "claude 'hello'" C-m
tmux send-keys -t $SESSION_NAME:0.2 "claude 'hello'" C-m
tmux send-keys -t $SESSION_NAME:0.3 "claude 'hello'" C-m

sleep 4

# Select theme: 1 (dark mode)
echo "  Selecting dark mode theme..."
tmux send-keys -t $SESSION_NAME:0.0 "1" C-m
tmux send-keys -t $SESSION_NAME:0.1 "1" C-m
tmux send-keys -t $SESSION_NAME:0.2 "1" C-m
tmux send-keys -t $SESSION_NAME:0.3 "1" C-m

sleep 2

# Accept prompt warning (Enter)
echo "  Accepting prompt warning..."
tmux send-keys -t $SESSION_NAME:0.0 "" C-m
tmux send-keys -t $SESSION_NAME:0.1 "" C-m
tmux send-keys -t $SESSION_NAME:0.2 "" C-m
tmux send-keys -t $SESSION_NAME:0.3 "" C-m

sleep 2

# Continue in this folder (Enter)
echo "  Confirming folder..."
tmux send-keys -t $SESSION_NAME:0.0 "" C-m
tmux send-keys -t $SESSION_NAME:0.1 "" C-m
tmux send-keys -t $SESSION_NAME:0.2 "" C-m
tmux send-keys -t $SESSION_NAME:0.3 "" C-m

sleep 2

# One more Enter for final confirmation
echo "  Final confirmation..."
tmux send-keys -t $SESSION_NAME:0.0 "" C-m
tmux send-keys -t $SESSION_NAME:0.1 "" C-m
tmux send-keys -t $SESSION_NAME:0.2 "" C-m
tmux send-keys -t $SESSION_NAME:0.3 "" C-m

sleep 5

# Wait for Claude to finish the 'hello' response, then exit
echo "  Waiting for setup to complete..."
tmux send-keys -t $SESSION_NAME:0.0 C-c
tmux send-keys -t $SESSION_NAME:0.1 C-c
tmux send-keys -t $SESSION_NAME:0.2 C-c
tmux send-keys -t $SESSION_NAME:0.3 C-c

sleep 2

echo "Starting Ralph loops..."
# Start Ralph in all panes
tmux send-keys -t $SESSION_NAME:0.0 "./afk-ralph.sh 20" C-m
tmux send-keys -t $SESSION_NAME:0.1 "./afk-ralph.sh 20" C-m
tmux send-keys -t $SESSION_NAME:0.2 "./afk-ralph.sh 20" C-m
tmux send-keys -t $SESSION_NAME:0.3 "./afk-ralph.sh 20" C-m

tmux select-layout -t $SESSION_NAME:0 tiled

echo ""
echo "=========================================="
echo "    Interactive Ralph Shells Ready! 🚀"
echo "=========================================="
echo ""
echo "Configuration complete - settings saved!"
echo "Ralph loops started in all 4 instances!"
echo ""
echo "You'll see Claude's full interactive output:"
echo "  - Thinking process with colors"
echo "  - Code generation with syntax highlighting"
echo "  - Command execution"
echo "  - Git commits"
echo ""
echo "tmux controls:"
echo "  Ctrl+b arrow-keys  - Navigate panes"
echo "  Ctrl+b z           - Zoom current pane"
echo "  Ctrl+b [           - Scroll mode (q to exit)"
echo "  Ctrl+b d           - Detach (keeps running)"
echo "  Ctrl+C             - Stop current Ralph loop"
echo ""
echo "Check progress anytime:"
echo "  ./claude-fleet.sh progress"
echo ""
echo "Attaching to session in 2 seconds..."
sleep 2

tmux attach -t $SESSION_NAME
