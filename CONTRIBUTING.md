# Contributing to VehicleBERT

Thank you for your interest in contributing to VehicleBERT! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected vs actual behavior
- Your environment (OS, Python version, PyTorch version)
- Any relevant logs or screenshots

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Check existing issues to avoid duplicates
- Clearly describe the enhancement and its benefits
- Provide examples of how it would work

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
3. **Test your changes**:
   ```bash
   pytest tests/
   ```
4. **Commit your changes**:
   - Use clear, descriptive commit messages
   - Reference related issues (e.g., "Fixes #123")
5. **Push to your fork** and submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/vehiclebert.git
cd vehiclebert

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions focused and concise
- Add comments for complex logic

### Code Formatting

We use Black for code formatting:

```bash
black src/ scripts/ tests/
```

### Linting

We use Flake8 for linting:

```bash
flake8 src/ scripts/ tests/
```

## Testing

- Write tests for all new features
- Maintain or improve test coverage
- Run tests before submitting PR:

```bash
pytest tests/ -v
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update examples if API changes
- Consider adding a Jupyter notebook for complex features

## Adding New Entity Types

If you want to add new entity types:

1. Update `VehicleBERTConfig.ENTITY_LABELS` in `src/model.py`
2. Add examples to `SyntheticDataGenerator` in `src/data_preparation.py`
3. Update the README to reflect new entity types
4. Add test cases for the new entities

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs when relevant

Example:
```
Add support for custom entity types

- Add custom_labels parameter to VehicleBERTPredictor
- Update documentation with examples
- Add tests for custom entity handling

Fixes #42
```

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in GitHub Discussions
- Contact the maintainers

Thank you for contributing to VehicleBERT! 🚗
