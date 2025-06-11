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

# Change to the project directory
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
log "Project root: $PROJECT_ROOT"

# Install Modal if not available
if ! python3 -m modal --version &> /dev/null; then
    log "Modal not found, installing..."
    pip install modal
    
    # Verify installation
    if ! python3 -m modal --version &> /dev/null; then
        log "Error: Failed to install modal. Please install it manually."
        exit 1
    fi
fi

log "modal version: $(python3 -m modal --version)"

# Set up Modal authentication using Buildkite secrets
log "Setting up Modal authentication from Buildkite secrets..."
MODAL_TOKEN_ID=$(buildkite-agent secret get modal_token_id)
MODAL_TOKEN_SECRET=$(buildkite-agent secret get modal_token_secret)

if [ -n "$MODAL_TOKEN_ID" ] && [ -n "$MODAL_TOKEN_SECRET" ]; then
    log "Retrieved Modal credentials from Buildkite secrets"
    python3 -m modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET" --profile buildkite-ci --activate --verify
    if [ $? -eq 0 ]; then
        log "Modal authentication successful"
    else
        log "Error: Failed to set Modal credentials"
        exit 1
    fi
else
    log "Error: Could not retrieve Modal credentials from Buildkite secrets."
    log "Please ensure 'modal_token_id' and 'modal_token_secret' secrets are set in Buildkite."
    exit 1
fi

# Check if the modal test file exists
MODAL_TEST_FILE="modal/test_gpu.py"
if [ ! -f "$MODAL_TEST_FILE" ]; then
    log "Error: Modal test file not found at $MODAL_TEST_FILE"
    exit 1
fi

log "Found Modal test file at $MODAL_TEST_FILE"

# Run the Modal test
log "Running Modal test..."
if python3 -m modal run modal/test_gpu.py; then
    TEST_EXIT_CODE=0
    log "Modal test completed successfully"
else
    TEST_EXIT_CODE=$?
    log "Error: Modal test failed with exit code: $TEST_EXIT_CODE"
fi

log "=== Test execution completed with exit code: $TEST_EXIT_CODE ==="

# Return the test exit code
exit $TEST_EXIT_CODE
