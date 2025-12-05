#!/bin/bash
# Helper script for common microtubule detection tasks

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Microtubule Detection Helper Script

Usage: ./run.sh [COMMAND] [OPTIONS]

Commands:
  setup               Install dependencies and set up environment
  inspect             Inspect dataset (requires --mrc-dir and --annotation-file)
  train               Start training (requires --mrc-dir and --annotation-file)
  inference           Run inference (requires --mrc-path or --mrc-dir and --model-path)
  evaluate            Evaluate model (requires --mrc-dir, --annotation-file, and --model-path)
  visualize           Visualize results (requires specific options based on mode)
  help                Show this help message

Common Options:
  --mrc-dir PATH              Directory containing MRC files
  --mrc-path PATH             Path to single MRC file (for inference)
  --annotation-file PATH      Path to annotation file
  --model-path PATH           Path to trained model
  --output-dir PATH           Output directory (default: output/default)
  --config PATH               Path to config file

Examples:
  ./run.sh setup
  ./run.sh inspect --mrc-dir data/mrc --annotation-file data/annotations.txt
  ./run.sh train --mrc-dir data/mrc --annotation-file data/annotations.txt
  ./run.sh train --config config/default.txt
  ./run.sh inference --mrc-path data/mrc/image.mrc --model-path output/best_model.pth
  ./run.sh evaluate --mrc-dir data/mrc --annotation-file data/annotations.txt --model-path output/best_model.pth

EOF
}

# Function to check if Python is available
check_python() {
    if ! command -v python &> /dev/null; then
        print_warning "python command not found, trying python3..."
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            echo "Error: Python not found. Please install Python 3.8+"
            exit 1
        fi
    else
        PYTHON_CMD="python"
    fi
    print_info "Using Python: $PYTHON_CMD"
}

# Setup command
cmd_setup() {
    print_info "Setting up environment..."
    
    check_python
    
    # Check if venv exists
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_info "Virtual environment already exists"
    fi
    
    # Activate venv and install dependencies
    print_info "Installing dependencies..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Setup complete!"
    print_info "To activate the environment, run: source venv/bin/activate"
}

# Inspect command
cmd_inspect() {
    check_python
    print_info "Inspecting dataset..."
    $PYTHON_CMD inspect_data.py "$@"
    print_success "Inspection complete!"
}

# Train command
cmd_train() {
    check_python
    print_info "Starting training..."
    $PYTHON_CMD train.py "$@"
    print_success "Training complete!"
}

# Inference command
cmd_inference() {
    check_python
    print_info "Running inference..."
    $PYTHON_CMD inference.py "$@"
    print_success "Inference complete!"
}

# Evaluate command
cmd_evaluate() {
    check_python
    print_info "Evaluating model..."
    $PYTHON_CMD evaluate.py "$@"
    print_success "Evaluation complete!"
}

# Visualize command
cmd_visualize() {
    check_python
    print_info "Creating visualization..."
    $PYTHON_CMD visualize.py "$@"
    print_success "Visualization complete!"
}

# Main script logic
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    COMMAND=$1
    shift  # Remove command from arguments
    
    case $COMMAND in
        setup)
            cmd_setup
            ;;
        inspect)
            cmd_inspect "$@"
            ;;
        train)
            cmd_train "$@"
            ;;
        inference)
            cmd_inference "$@"
            ;;
        evaluate)
            cmd_evaluate "$@"
            ;;
        visualize)
            cmd_visualize "$@"
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            echo "Error: Unknown command '$COMMAND'"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
