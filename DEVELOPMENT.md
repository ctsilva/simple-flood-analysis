# Development Guide

This guide provides comprehensive information for developers working on the flood analysis toolkit.

## Project Structure

```
simple-flood-analysis/
├── CLAUDE.md                     # Claude AI guidance
├── README.md                     # Project overview
├── DEVELOPMENT.md                # This development guide
├── requirements.txt              # Python dependencies
├── pytest.ini                   # Test configuration
├── 
├── config.py                     # Configuration management
├── logger.py                     # Logging utilities  
├── validation.py                 # Input validation functions
├── 
├── elevation_data_analyzer.py    # Core DEM analysis
├── flood_analysis.py             # Pure Python flood analysis
├── nyc_data_clipping.py          # Geospatial data clipping
├── raster_clipping_methods.py    # Memory-efficient raster processing
├── nyc_data_download_guide2.py   # Data acquisition
├── 
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Pytest configuration and fixtures
│   ├── test_validation.py        # Validation tests
│   └── test_config.py            # Configuration tests
├── 
└── examples/                     # Usage examples
    ├── basic_flood_analysis.py   # Basic examples
    └── advanced_flood_analysis.py # Advanced examples
```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- Recommended: Virtual environment (venv, conda, etc.)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/simple-flood-analysis.git
   cd simple-flood-analysis
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies:**
   ```bash
   # Testing and code quality tools
   pip install pytest pytest-cov pytest-timeout
   pip install black isort flake8 mypy
   ```

## Code Style and Standards

### Python Style

- **PEP 8**: Follow Python style guidelines
- **Type Hints**: Use comprehensive type annotations
- **Docstrings**: Use Google-style docstrings
- **Line Length**: Maximum 88 characters (Black default)

### Code Formatting

Use **Black** for code formatting:
```bash
black .
```

Use **isort** for import sorting:
```bash
isort .
```

### Linting

Use **flake8** for linting:
```bash
flake8 .
```

Use **mypy** for type checking:
```bash
mypy .
```

## Testing

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=. --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_validation.py
```

Run tests with specific markers:
```bash
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run only integration tests
```

### Writing Tests

1. **Test File Naming**: `test_*.py`
2. **Test Function Naming**: `test_*`
3. **Test Class Naming**: `Test*`

Example test structure:
```python
import pytest
from your_module import your_function

class TestYourFunction:
    def test_valid_input(self):
        """Test function with valid input."""
        result = your_function("valid_input")
        assert result == expected_result
    
    def test_invalid_input(self):
        """Test function with invalid input."""
        with pytest.raises(ValueError):
            your_function("invalid_input")
    
    @pytest.mark.slow
    def test_large_dataset(self):
        """Test function with large dataset (marked as slow)."""
        # Long-running test
        pass
```

### Test Fixtures

Common fixtures are available in `tests/conftest.py`:
- `temp_directory`: Temporary directory for test files
- `sample_dem_data`: Sample DEM data for testing
- `mock_rasterio_dataset`: Mocked rasterio dataset
- `sample_roads_geojson`: Sample road network data

## Architecture Overview

### Core Components

1. **Configuration (`config.py`)**
   - Centralized configuration management
   - Environment variable handling
   - Default values and constants

2. **Validation (`validation.py`)**
   - Input validation functions
   - File existence and format checks
   - Parameter range validation
   - Custom exception types

3. **Logging (`logger.py`)**
   - Centralized logging configuration
   - Progress tracking utilities
   - Memory usage monitoring

4. **Analysis Modules**
   - `elevation_data_analyzer.py`: DEM analysis and visualization
   - `flood_analysis.py`: Pure Python flood risk analysis
   - `nyc_data_clipping.py`: Geospatial data clipping
   - `raster_clipping_methods.py`: Memory-efficient raster processing

### Design Patterns

1. **Configuration Pattern**
   - All configurable values in `config.py`
   - Environment variable overrides
   - Type-safe configuration access

2. **Validation Pattern**
   - Input validation at entry points
   - Custom exception hierarchy
   - Comprehensive error messages

3. **Logging Pattern**
   - Structured logging throughout
   - Configurable log levels
   - Progress tracking for long operations

4. **Error Handling Pattern**
   - Custom exception types
   - Graceful degradation
   - User-friendly error messages

## Adding New Features

### 1. Planning

1. **Create Issue**: Document the feature requirement
2. **Design**: Plan the implementation approach
3. **API Design**: Define function signatures and interfaces
4. **Testing Strategy**: Plan test coverage

### 2. Implementation

1. **Create Branch**: Use descriptive branch names
   ```bash
   git checkout -b feature/new-analysis-method
   ```

2. **Add Configuration**: Update `config.py` if needed
   ```python
   # Add new configuration parameters
   NEW_FEATURE_DEFAULT_VALUE = 42
   NEW_FEATURE_THRESHOLDS = {'low': 0.1, 'high': 0.9}
   ```

3. **Add Validation**: Update `validation.py` if needed
   ```python
   def validate_new_parameter(value: float) -> float:
       """Validate new parameter value."""
       return validate_numeric_parameter(
           value, "new_parameter", min_value=0, max_value=100
       )
   ```

4. **Implement Feature**: Add core functionality
   ```python
   class NewAnalysisMethod:
       def __init__(self, config_value: float = None):
           self.logger = get_logger(self.__class__.__name__)
           self.config_value = config_value or config.NEW_FEATURE_DEFAULT_VALUE
       
       def analyze(self, data: np.ndarray) -> Dict[str, float]:
           """Perform new analysis."""
           self.logger.info("Starting new analysis")
           # Implementation here
           return results
   ```

5. **Add Tests**: Create comprehensive tests
   ```python
   class TestNewAnalysisMethod:
       def test_analyze_valid_data(self):
           """Test analysis with valid data."""
           method = NewAnalysisMethod()
           result = method.analyze(sample_data)
           assert isinstance(result, dict)
           assert 'metric1' in result
   ```

6. **Add Documentation**: Update docstrings and examples

### 3. Code Review

1. **Self Review**: Check your code before creating PR
2. **Run Tests**: Ensure all tests pass
3. **Check Coverage**: Maintain or improve test coverage
4. **Format Code**: Run Black and isort
5. **Create PR**: Use descriptive PR description

## Configuration Management

### Environment Variables

Override default settings using environment variables:

```bash
# Data directories
export FLOOD_ANALYSIS_DATA_DIR="/path/to/data"
export FLOOD_ANALYSIS_OUTPUT_DIR="/path/to/output"

# Performance settings
export FLOOD_ANALYSIS_MAX_MEMORY_MB="1024"

# Logging
export FLOOD_ANALYSIS_LOG_LEVEL="DEBUG"
export FLOOD_ANALYSIS_LOG_FILE="/path/to/logfile.log"

# Debug mode
export FLOOD_ANALYSIS_DEBUG="1"
```

### Adding New Configuration

1. **Add to Config Class**:
   ```python
   class FloodAnalysisConfig:
       NEW_SETTING = "default_value"
       
       @classmethod
       def get_new_setting(cls, fallback=None):
           return fallback or cls.NEW_SETTING
   ```

2. **Add Environment Support**:
   ```python
   class EnvironmentConfig:
       @staticmethod
       def get_new_setting():
           return os.getenv('FLOOD_ANALYSIS_NEW_SETTING')
   ```

3. **Add Tests**:
   ```python
   def test_new_setting_default():
       result = FloodAnalysisConfig.get_new_setting()
       assert result == FloodAnalysisConfig.NEW_SETTING
   ```

## Error Handling

### Exception Hierarchy

```python
ValidationError
├── FileValidationError      # File-related validation errors
├── DataValidationError      # Data integrity errors  
└── ParameterValidationError # Parameter validation errors
```

### Error Handling Patterns

1. **Input Validation**:
   ```python
   def process_data(input_path: Path, threshold: float):
       # Validate inputs at entry point
       validated_path = validate_file_path(input_path)
       validated_threshold = validate_numeric_parameter(
           threshold, "threshold", min_value=0, max_value=100
       )
       # Process with validated inputs
   ```

2. **Graceful Degradation**:
   ```python
   try:
       result = complex_operation()
       return result
   except ComplexOperationError as e:
       logger.warning(f"Complex operation failed: {e}")
       return simple_fallback_operation()
   ```

3. **User-Friendly Messages**:
   ```python
   try:
       load_data(file_path)
   except FileValidationError as e:
       print(f"❌ {e}")
       print("💡 Please check that the file exists and is readable")
       return False
   ```

## Performance Considerations

### Memory Management

1. **Large Raster Processing**: Use windowing and chunking
2. **Visualization**: Sample large datasets
3. **Memory Monitoring**: Use built-in memory tracking

### Performance Testing

Mark slow tests appropriately:
```python
@pytest.mark.slow
def test_large_dataset_processing():
    """Test processing of large dataset."""
    # Test with large data
    pass
```

Run performance tests:
```bash
pytest -m slow
```

## Documentation

### Docstring Format

Use Google-style docstrings:
```python
def analyze_flood_risk(elevation_data: np.ndarray, 
                      water_level: float) -> Dict[str, float]:
    """Analyze flood risk for given elevation data.
    
    This function computes flood risk metrics including affected area,
    maximum flood depth, and risk distribution.
    
    Args:
        elevation_data: 2D array of elevation values in feet
        water_level: Water level threshold in feet above sea level
        
    Returns:
        Dictionary containing flood risk metrics:
        - 'affected_area_pct': Percentage of area affected
        - 'max_depth': Maximum flood depth in feet
        - 'mean_depth': Mean flood depth in affected areas
        
    Raises:
        ValueError: If elevation_data is empty or water_level is invalid
        
    Examples:
        >>> elevation = np.array([[1, 5, 10], [2, 8, 15]])
        >>> metrics = analyze_flood_risk(elevation, 6.0)
        >>> print(f"Affected area: {metrics['affected_area_pct']:.1f}%")
        Affected area: 66.7%
    """
```

### Adding Examples

1. **Simple Examples**: Add to `examples/basic_flood_analysis.py`
2. **Advanced Examples**: Add to `examples/advanced_flood_analysis.py`
3. **Docstring Examples**: Include in function docstrings
4. **README Examples**: Add to main README.md

## Release Process

### Version Numbering

Use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update Version**: Update version in relevant files
2. **Update CHANGELOG**: Document changes
3. **Run Full Test Suite**: Ensure all tests pass
4. **Update Documentation**: Ensure docs are current
5. **Create Release**: Tag and create GitHub release
6. **Update Dependencies**: Check for dependency updates

## Troubleshooting

### Common Issues

1. **Import Errors**: Check Python path and virtual environment
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Test Failures**: Check test environment and fixtures
4. **Memory Issues**: Reduce dataset size or increase memory limits

### Getting Help

1. **Check Issues**: Look for existing GitHub issues
2. **Check Documentation**: Review this guide and README
3. **Create Issue**: Report bugs or request features
4. **Contact Maintainers**: Use GitHub discussions

## Contributing

1. **Fork Repository**: Create personal fork
2. **Create Branch**: Use descriptive branch names
3. **Make Changes**: Follow development guidelines
4. **Add Tests**: Ensure comprehensive test coverage
5. **Update Documentation**: Keep docs current
6. **Create Pull Request**: Use clear PR description
7. **Address Feedback**: Respond to review comments

Thank you for contributing to the flood analysis toolkit! 🌊