# Contributing to ResearchAgent

Thank you for your interest in contributing to ResearchAgent! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/ResearchAgent.git
   cd ResearchAgent
   ```
3. **Add the upstream repository**:
   ```bash
   git remote add upstream https://github.com/ashaduzzaman-sarker/ResearchAgent.git
   ```

## Development Setup

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Virtual environment tool (venv recommended)

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Install pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

### Branch Naming Convention

- `feature/description` - For new features
- `bugfix/description` - For bug fixes
- `docs/description` - For documentation updates
- `refactor/description` - For code refactoring
- `test/description` - For test additions/updates

### Development Workflow

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write clean, readable code
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add description of your changes"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Adding or updating tests
   - `refactor:` - Code refactoring
   - `style:` - Code style changes (formatting, etc.)
   - `chore:` - Maintenance tasks

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_extraction.py

# Run specific test
pytest tests/test_data_extraction.py::TestDataExtraction::test_load_config_success
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions/methods as `test_*`
- Use fixtures from `conftest.py`
- Aim for high test coverage (>80%)

Example:
```python
def test_your_feature():
    # Arrange
    input_data = {"key": "value"}
    
    # Act
    result = your_function(input_data)
    
    # Assert
    assert result == expected_output
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for modules, classes, and functions

### Formatting Tools

The project uses the following tools (run automatically with pre-commit):

```bash
# Format code with Black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Lint code with Ruff
ruff check src/ tests/

# Type check with mypy
mypy src/
```

### Documentation Style

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When param1 is empty.
    
    Example:
        >>> function_name("test", 42)
        True
    """
    pass
```

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template

### Pull Request Guidelines

- **Title**: Clear, concise description of changes
- **Description**: 
  - Explain what changes you made and why
  - Link any related issues
  - Include screenshots for UI changes
  - List any breaking changes

- **Checklist**:
  - [ ] Code follows the project's style guidelines
  - [ ] Self-review completed
  - [ ] Comments added for complex code
  - [ ] Documentation updated
  - [ ] Tests added/updated
  - [ ] All tests pass
  - [ ] No new warnings generated

### Review Process

1. Automated CI/CD checks will run
2. Maintainers will review your code
3. Address any feedback or requested changes
4. Once approved, your PR will be merged

## Questions or Issues?

- **Questions**: Open a discussion on GitHub Discussions
- **Bugs**: Open an issue with the bug template
- **Feature Requests**: Open an issue with the feature request template

## License

By contributing to ResearchAgent, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ResearchAgent! 🎉
