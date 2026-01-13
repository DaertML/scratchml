#!/bin/bash
# Save as: claude-fleet.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

show_help() {
    cat << EOF
Claude Fleet Manager (Ralph Loop)

Usage: $0 [command]

Commands:
    setup [template]    Initialize Ralph project (optional: todo-app, web-app, api-service)
    start              Start all Claude Ralph instances (interactive shells)
    run                Start all Ralph loops automatically
    stop               Stop all Claude instances
    restart            Restart all Claude instances
    status             Show status of all instances
    logs [N]           Show logs for instance N (or all)
    follow [N]         Follow logs in real-time
    progress           Show current progress
    snapshot           Save current tmux output to files
    tmux               Launch tmux multi-window session (interactive)
    clean              Stop and remove all containers
    reset              Clean everything and reset project
    templates          List available project templates

Examples:
    $0 setup todo-app     # Setup with todo template
    $0 start              # Start interactive shells (manual control)
    $0 run                # Start automatic Ralph loops
    $0 snapshot           # Save all pane outputs
    $0 progress           # Check progress

EOF
}

setup_ralph() {
    if [ ! -f setup-ralph.sh ]; then
        echo "Error: setup-ralph.sh not found"
        exit 1
    fi
    ./setup-ralph.sh "$1"
}

list_templates() {
    echo "Available Project Templates:"
    echo "============================"
    echo ""
    if [ -d project-templates ]; then
        for template in project-templates/*-PRD.md; do
            if [ -f "$template" ]; then
                template_name=$(basename "$template" | sed 's/-PRD.md//')
                echo "  $template_name"
                echo "    Usage: ./claude-fleet.sh setup $template_name"
                echo ""
            fi
        done
    else
        echo "No project-templates directory found."
    fi
}

start_instances() {
    if [ ! -f shared-project/PRD.md ]; then
        echo "⚠️  Error: PRD.md not found in shared-project/"
        echo ""
        echo "Create it first:"
        echo "  ./claude-fleet.sh setup todo-app    # Use a template"
        echo "  ./claude-fleet.sh templates          # See all templates"
        echo ""
        exit 1
    fi
    
    if [ -f docker-compose.yaml ]; then
        echo "Removing conflicting docker-compose.yaml..."
        rm docker-compose.yaml
    fi
    
    echo "Starting all Claude instances in INTERACTIVE mode..."
    echo "You'll have direct shell access in each container."
    echo ""
    sudo docker-compose up -d
    
    sleep 2
    
    echo ""
    echo "✓ All instances running!"
    echo ""
    echo "Opening tmux with interactive shells..."
    sleep 1
    
    ./launch-tmux.sh
}

run_ralph_loops() {
    if [ ! -f shared-project/PRD.md ]; then
        echo "⚠️  Error: PRD.md not found in shared-project/"
        exit 1
    fi
    
    echo "Starting Ralph loops in all containers..."
    
    for i in 1 2 3 4; do
        docker exec -d claude-ai-$i bash -c "cd /project && ./afk-ralph.sh 20"
        echo "✓ Started Ralph loop in instance $i"
    done
    
    echo ""
    echo "All Ralph loops started!"
    echo "Watch with: ./claude-fleet.sh tmux"
}

stop_instances() {
    sudo docker-compose down
    echo "✓ Stopped."
}

restart_instances() {
    sudo docker-compose restart
    echo "✓ Restarted."
}

show_status() {
    docker-compose ps
}

show_logs() {
    if [ -z "$1" ]; then
        docker-compose logs --tail=100
    else
        docker-compose logs --tail=100 "claude-lab-$1"
    fi
}

follow_logs() {
    if [ -z "$1" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "claude-lab-$1"
    fi
}

snapshot_tmux() {
    SESSION_NAME="claude-ralph"
    
    if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "⚠️  Tmux session '$SESSION_NAME' not found. Start it first with:"
        echo "  ./claude-fleet.sh start"
        exit 1
    fi
    
    SNAPSHOT_DIR="snapshots/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$SNAPSHOT_DIR"
    
    echo "Capturing tmux panes..."
    echo ""
    
    for i in 0 1 2 3; do
        instance=$((i + 1))
        output_file="$SNAPSHOT_DIR/instance-${instance}.txt"
        
        tmux capture-pane -t "${SESSION_NAME}:0.${i}" -p -S -5000 > "$output_file"
        echo "✓ Saved instance $instance to: $output_file"
    done
    
    echo ""
    echo "=========================================="
    echo "✓ Snapshot saved to: $SNAPSHOT_DIR"
    echo "=========================================="
}

show_progress() {
    echo "Progress:"
    echo "========="
    if [ -f shared-project/progress.txt ]; then
        cat shared-project/progress.txt
    else
        echo "No progress yet."
    fi
    echo ""
    echo "PRD Status:"
    echo "==========="
    if [ -f shared-project/PRD.md ]; then
        completed=$(grep -E '^\- \[x\]' shared-project/PRD.md | wc -l)
        remaining=$(grep -E '^\- \[ \]' shared-project/PRD.md | wc -l)
        total=$((completed + remaining))
        echo "Completed: $completed/$total"
        echo "Remaining: $remaining/$total"
        
        if [ $remaining -gt 0 ]; then
            echo ""
            echo "Next tasks:"
            grep -E '^\- \[ \]' shared-project/PRD.md | head -5
        fi
    fi
    
    echo ""
    echo "Git commits:"
    echo "============"
    if [ -d shared-project/.git ]; then
        cd shared-project
        git log --oneline --all --graph -10 2>/dev/null || echo "No commits yet"
        cd ..
    fi
}

launch_tmux() {
    if [ ! -f launch-tmux.sh ]; then
        echo "Error: launch-tmux.sh not found"
        exit 1
    fi
    ./launch-tmux.sh
}

clean_all() {
    docker-compose down -v
    echo "✓ Cleaned."
}

reset_project() {
    read -p "Reset project? This will delete shared-project/ (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        docker-compose down -v
        rm -rf shared-project
        echo "✓ Reset complete. Run './claude-fleet.sh setup <template>' to start fresh."
    else
        echo "Reset cancelled."
    fi
}

case "$1" in
    setup) setup_ralph "$2" ;;
    templates) list_templates ;;
    start) start_instances ;;
    run) run_ralph_loops ;;
    stop) stop_instances ;;
    restart) restart_instances ;;
    status) show_status ;;
    logs) show_logs "$2" ;;
    follow) follow_logs "$2" ;;
    progress) show_progress ;;
    snapshot) snapshot_tmux ;;
    tmux) launch_tmux ;;
    clean) clean_all ;;
    reset) reset_project ;;
    *) show_help ;;
esac
