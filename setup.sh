#!/bin/bash

# NBA Betting Predictions System - Setup Script
# This script automates the setup process for the prediction system

set -e  # Exit on error

echo "=========================================="
echo "NBA Betting Predictions System - Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check Python version
echo "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet
print_success "pip upgraded"

# Install requirements
echo ""
echo "Installing Python dependencies..."
echo "(This may take a few minutes...)"
pip install -r requirements.txt --quiet
print_success "All dependencies installed"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file for configuration..."
    cat > .env << 'EOF'
# The Odds API Configuration
# Sign up at https://the-odds-api.com/ to get your API key
ODDS_API_KEY=your_api_key_here

# Optional: Configure which sportsbooks to use
# ODDS_REGIONS=us
# ODDS_MARKETS=player_points,player_rebounds,player_assists

# Optional: Set confidence threshold for predictions (0.0-1.0)
# MIN_CONFIDENCE=0.6
EOF
    print_success ".env file created"
    print_warning "Please edit .env and add your ODDS_API_KEY"
else
    print_warning ".env file already exists"
fi

# Check if models directory exists
echo ""
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
    print_success "Models directory created"
fi

# Check if data directories exist
if [ ! -d "data/processed" ]; then
    echo "Creating data directories..."
    mkdir -p data/raw data/processed
    print_success "Data directories created"
fi

# Check if models are trained
echo ""
echo "Checking for trained models..."
MODEL_FILES=("logistic_regression_model.pkl" "random_forest_model.pkl" "xgboost_model.pkl" "feature_columns.pkl")
MODELS_FOUND=0

for model in "${MODEL_FILES[@]}"; do
    if [ -f "models/$model" ]; then
        ((MODELS_FOUND++))
    fi
done

if [ $MODELS_FOUND -eq ${#MODEL_FILES[@]} ]; then
    print_success "All trained models found ($MODELS_FOUND/${#MODEL_FILES[@]})"
elif [ $MODELS_FOUND -gt 0 ]; then
    print_warning "Some models found ($MODELS_FOUND/${#MODEL_FILES[@]})"
    echo "   You may need to retrain missing models"
else
    print_warning "No trained models found"
    echo "   Run the training scripts in src/ to train the models"
fi

# Check for historical data
echo ""
echo "Checking for historical data..."
if [ -f "data/processed/engineered_features.csv" ]; then
    print_success "Historical feature data found"
else
    print_warning "No historical data found"
    echo "   You'll need to run the data processing pipeline first"
    echo "   See src/data_cleaning.py and src/feature_engineering.py"
fi

# Final instructions
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Get your API key:"
echo "   → Visit https://the-odds-api.com/"
echo "   → Sign up for a free account (500 requests/month)"
echo "   → Copy your API key"
echo ""
echo "2. Configure your environment:"
echo "   → Edit .env file and add your API key:"
echo "     ODDS_API_KEY='your_actual_key_here'"
echo ""
echo "3. Test your setup:"
echo "   → Run: source venv/bin/activate"
echo "   → Run: python src/odds_api_client.py"
echo ""
echo "4. Generate daily predictions:"
echo "   → Run: python src/daily_predictions.py"
echo ""
echo "For more information, see SETUP.md"
echo ""

# Check if API key is set
if [ -f ".env" ]; then
    source .env
    if [ "$ODDS_API_KEY" = "your_api_key_here" ] || [ -z "$ODDS_API_KEY" ]; then
        print_warning "Don't forget to set your ODDS_API_KEY in .env!"
    else
        print_success "API key appears to be configured"
    fi
fi

echo ""
echo "Happy predicting! 🏀"
