"""
Custom exceptions for dob-sob platform.

Provides specific exception classes for different error scenarios
to enable better error handling and debugging.
"""


class DobSobError(Exception):
    """Base exception for all dob-sob related errors."""
    pass


class ConfigurationError(DobSobError):
    """Raised when there are configuration-related errors."""
    pass


class DataAcquisitionError(DobSobError):
    """Base class for data acquisition errors."""
    pass


class DatasetNotFoundError(DataAcquisitionError):
    """Raised when a requested dataset cannot be found."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        super().__init__(f"Dataset '{dataset_name}' not found")


class APIConnectionError(DataAcquisitionError):
    """Raised when unable to connect to NYC Open Data API."""
    
    def __init__(self, url: str, status_code: int = None, message: str = None):
        self.url = url
        self.status_code = status_code
        if message is None:
            if status_code:
                message = f"API connection failed for {url} with status {status_code}"
            else:
                message = f"API connection failed for {url}"
        super().__init__(message)


class DownloadError(DataAcquisitionError):
    """Raised when data download fails."""
    
    def __init__(self, message: str):
        super().__init__(message)


class DataValidationError(DataAcquisitionError):
    """Raised when downloaded data fails validation."""
    
    def __init__(self, dataset_name: str, validation_type: str, details: str = None):
        self.dataset_name = dataset_name
        self.validation_type = validation_type
        self.details = details
        message = f"Data validation failed for '{dataset_name}' ({validation_type})"
        if details:
            message += f": {details}"
        super().__init__(message)


class DatabaseError(DobSobError):
    """Base class for database-related errors."""
    pass


class Neo4jConnectionError(DatabaseError):
    """Raised when unable to connect to Neo4j database."""
    
    def __init__(self, uri: str, message: str = None):
        self.uri = uri
        if message is None:
            message = f"Failed to connect to Neo4j at {uri}"
        super().__init__(message)


class QueryExecutionError(DatabaseError):
    """Raised when a database query fails to execute."""
    
    def __init__(self, query: str, error_message: str):
        self.query = query
        self.error_message = error_message
        super().__init__(f"Query execution failed: {error_message}")


class FraudDetectionError(DobSobError):
    """Base class for fraud detection errors."""
    pass


class AlgorithmError(FraudDetectionError):
    """Raised when a fraud detection algorithm fails."""
    
    def __init__(self, algorithm_name: str, reason: str):
        self.algorithm_name = algorithm_name
        self.reason = reason
        super().__init__(f"Algorithm '{algorithm_name}' failed: {reason}")


class CommunityDetectionError(FraudDetectionError):
    """Raised when community detection algorithms fail."""
    
    def __init__(self, algorithm: str, dataset: str, reason: str):
        self.algorithm = algorithm
        self.dataset = dataset
        self.reason = reason
        super().__init__(
            f"Community detection failed for algorithm '{algorithm}' "
            f"on dataset '{dataset}': {reason}"
        )


class GraphAnalysisError(FraudDetectionError):
    """Raised when graph analysis operations fail."""
    
    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Graph analysis operation '{operation}' failed: {reason}")


class WebInterfaceError(DobSobError):
    """Base class for web interface errors."""
    pass


class DashboardError(WebInterfaceError):
    """Raised when dashboard operations fail."""
    pass


class CLIError(DobSobError):
    """Base class for CLI-related errors."""
    pass


class CommandExecutionError(CLIError):
    """Raised when a CLI command fails to execute."""
    
    def __init__(self, command: str, reason: str):
        self.command = command
        self.reason = reason
        super().__init__(f"Command '{command}' failed: {reason}")


class InvalidArgumentError(CLIError):
    """Raised when invalid arguments are provided to CLI commands."""
    
    def __init__(self, argument: str, value: str, reason: str = None):
        self.argument = argument
        self.value = value
        self.reason = reason
        message = f"Invalid argument '{argument}' with value '{value}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)