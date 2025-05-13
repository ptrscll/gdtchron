# How to contribute to GDTchron
We welcome and encourage you to contribute to GDTchron! This document is intended as a guide for starting out if you aren't sure how to begin.

## Bugs or feature reqeusts
If you would like to report a bug or ask other developers to implement a feature, please open a GitHub issue and describe the bug or request there.

## Code development
There are multiple ways to develop a Python package, and our goal with GDTchron is to reduce barriers to contribution as much as possible. At the same time, we want the code to be as consistent and readable as possible for users, so we have a suggested workflow for designing unit tests for changes and keeping formatting consistent.

### 1. Fork and clone the repository
Fork the repository to your own GitHub account, then clone the repository locally
```
git clone https://github.com/<your-username>/gdtchron.git
cd gdtchron
```
Replace `<your-username>` with your GitHub username.
### 2. Install Poetry
Poetry is a Python packaging tool that makes it easier to handle dependencies, testing, and linting. See here for multiple methods of installing it: https://python-poetry.org/docs/#installation

We recommend using the `pipx` method to install Poetry globally on your system:
```
pipx install poetry
```
You can also install and modify GDTchron without Poetry (e.g., using pip and/or Conda), but you will then need to manually install Pytest and Ruff in order to do local testing and linting

### 3. Build the package using Poetry
Once Poetry is installed, simply install GDTchron with the command:
```
poetry install
```
This will install the dependencies for the package in a virtual environment, along with Pytest for testing and Ruff for linting.

### 4. Make changes to the code
Make your changes to the code, making sure to run tests and to lint your code.

#### Tests
Every function within a module needs a corresponding test to ensure that it is operating as intended when changes are made to the code and across Python versions. Each module should have a `.py` file in the `tests` directory with the name after the format `test_module1.py`. Within the file, a test function needs to be defined for each function in the module. The test function should call the function and check that the output is as intended using `assert` statements:
```
from gdtchron.subpackage import module1

def test_function():
    output = module1.function(input)
    assert output == correct_answer
```
To check that all tests are passing, run Pytest via Poetry using the following command:
```
poetry run pytest
```

#### Linting
To ensure that the code is readable, we use Ruff as a linter to maintain consistent code style. To check if your code is passing the rules set by Ruff, run the following command:
```
poetry run ruff check .
```

### 4. Commit and pull request
Commit and push your code changes to GitHub. Use a short but descriptive present tense message to indicate what your commit does:
```
git add file.py
git commit -m "Add feature to code."
git push
```
On GitHub, make a pull request to request merging your changes into the `main` branch. The pull request will automatically run Pytest and Ruff to check the code, and we may ask you to squash multiple commits using rebase to keep the commit history organized.

Please feel free to reach out if you have any questions about these steps. Our goal is to make this process as open and transparent as possible while keeping the code manageable.

