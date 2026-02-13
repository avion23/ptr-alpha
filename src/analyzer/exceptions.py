class AnalyzerError(Exception):
    pass

class DataSourceError(AnalyzerError):
    pass

class ParsingError(AnalyzerError):
    pass

class AnalysisError(AnalyzerError):
    pass

class ConfigurationError(AnalyzerError):
    pass