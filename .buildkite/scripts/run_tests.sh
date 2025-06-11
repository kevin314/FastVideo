#!/bin/bash
# Change to exit on error but continue on command failures
set -uo pipefail

# Function to log with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if we're already running as buildkite-agent
if [[ "$(whoami)" != "buildkite-agent" ]]; then
    log "Switching to buildkite-agent user..."
    # Re-run this script as the buildkite-agent user
    exec sudo -i -u buildkite-agent "$0" "$@"
    # The exec command replaces the current process, so no code after this will run
    # if the switch is successful
fi

log "Running as $(whoami)"
log "=== Starting Modal test execution ==="

# Verify modal is installed and working
if ! command -v modal &> /dev/null; then
    log "Error: modal is not installed. Please install it first."
    exit 1
fi

log "modal version: $(modal --version)"

# Check if modal is authenticated
log "Checking modal authentication..."
if ! modal token list &> /dev/null; then
    log "Error: modal is not authenticated. Please run 'modal token new' first."
    exit 1
fi

# Change to the project directory
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
log "Project root: $PROJECT_ROOT"

# Check if the modal test file exists
MODAL_TEST_FILE="modal/test_gpu.py"
if [ ! -f "$MODAL_TEST_FILE" ]; then
    log "Error: Modal test file not found at $MODAL_TEST_FILE"
    exit 1
fi

log "Found Modal test file at $MODAL_TEST_FILE"

# Run the Modal test
log "Running Modal test..."
if modal run modal/test_gpu.py; then
    TEST_EXIT_CODE=0
    log "Modal test completed successfully"
else
    TEST_EXIT_CODE=$?
    log "Error: Modal test failed with exit code: $TEST_EXIT_CODE"
fi

log "=== Test execution completed with exit code: $TEST_EXIT_CODE ==="

# Return the test exit code
exit $TEST_EXIT_CODE
