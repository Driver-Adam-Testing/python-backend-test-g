# Claude Code Assistant Guidelines

## Clean Code Principles

### 1. Self-Documenting Code
- **Clarity of intent is paramount** - code should reveal its purpose through structure and naming, not comments
- Well-structured code helps both humans and AI agents understand and extend the codebase correctly
- Use descriptive function and variable names instead of verbose comments
- Break complex logic into small, well-named helper functions that express intent
- Each function should do one thing well, making it easily testable
- Avoid self-evident comments and docstrings that just restate what the code does
- Comments should explain *why*, not *what* - if the code is clear, no comment is needed
- Example of bad docstring: `"""Get user by ID."""` for a function named `get_user_by_id`
- Example of good comment: Explaining business rules like "public visibility takes precedence over internal"

### 2. Function Naming
- Use clear, action-oriented names for functions (e.g., `fetch_nodes_with_descriptions`, `_build_tree_from_nodes`)
- Private functions should start with underscore
- Function names should describe what they do, not how they do it

### 3. Avoid Code Smells
- Replace inline comments with well-named functions
- Avoid deeply nested logic - extract to helper functions
- Keep functions small (typically under 10-15 lines)
- Dont repeat yourself (DRY principle)
- Extract duplicated logic into helper functions
- When refactoring, look for opportunities to consolidate similar code patterns

### 4. Data Structure Oriented Design
- **Prefer structured data over raw dicts** - define your data shapes explicitly (e.g., Pydantic models)
- This catches errors at development time, provides IDE autocomplete, validation, and self-documenting code
- When parsing external data (API responses, config files), convert to structured types at the boundary
- Use `dict` only when keys are truly dynamic (e.g., a mapping of user IDs to objects)
- Example of bad: `def process_user(user: dict[str, Any])` - unclear what fields exist
- Example of good: `def process_user(user: User)` where `User` is a Pydantic model

### 5. Type Hints
- Always include type hints for function parameters and return values
- Use specific types over generic ones (e.g., `dict[str, Any]` instead of just `dict`)
- Use `Literal` or enum types for constrained string values (e.g., `Literal["private", "internal", "public"]`)
- Prefer defined types/schemas over raw strings for API parameters

### 6. Error Handling
- try/except blocks should be as narrow as possible.
- Huge try/except blocks that catch all exceptions are bad practice.
- Use descriptive error messages
- Handle edge cases explicitly

### 7. Testing
- Write unit tests when possible
- Write integration tests but only if they *really* test functionality. I.e. they dont just use mocks for *everything* and are therefore hard to maintain
- DO NOT write tests that are a huge maintenance burden unless they are high value

### 8. Misc
- For all application-critical network requests, use a retry mechanism, such as retry_with_exponential_backoff with appropriate parameters. Use discresion.

## Development Workflow

### Running Python Commands
- **Always use uv** to run Python commands in the backend
- Use `cd backend && uv run python -m pytest ...` instead of `python -m pytest ...`
- Use `cd backend && uv run python ...` instead of `python ...`
- This ensures you are using the correct environment with all dependencies

