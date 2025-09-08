## How to Contribute

### 1. Fork and Clone

- Fork the repository on GitHub
- Clone your fork locally:
  ```bash
  git clone https://github.com/{yourusername}/omop_mcp.git
  cd omop_mcp
  ```

### 2. Development Setup

- Install development dependencies:
  ```bash
  uv sync --extra dev
  ```
- Install pre-commit hooks:
  ```bash
  uv run pre-commit install
  ```
  Pre-commit hooks will automatically handle code formatting

### 3. Make Changes

- Create a feature branch:
  ```bash
  git checkout -b feature/{your-feature-name}
  ```

### 4. Submit Pull Request

- Push your changes to your fork
- Create a pull request with clear description of changes
