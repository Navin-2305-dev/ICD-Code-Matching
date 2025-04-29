# File: exceptions.py
class DatabaseSetupError(Exception):
    """Raised when setting up the FTS5 table or triggers fails."""
    pass

class FTSQueryError(Exception):
    """Raised when an FTS query execution fails."""
    pass

class ICDTitleRetrievalError(Exception):
    """Raised when retrieving an ICD title from the database fails."""
    pass

class EmbedderLoadError(Exception):
    """Raised when loading the sentence transformer model fails."""
    pass

class EmbeddingGenerationError(Exception):
    """Raised when generating embeddings for text fails."""
    pass

class SimilarityCalculationError(Exception):
    """Raised when calculating similarity scores between embeddings fails."""
    pass

class MistralQueryError(Exception):
    """Raised when querying the Mistral model fails after retries."""
    pass

class TextPreprocessingError(Exception):
    """Raised when preprocessing text fails."""
    pass

class ConditionExtractionError(Exception):
    """Raised when extracting conditions from patient text fails."""
    pass

class TrainingDataLoadError(Exception):
    """Raised when loading the training dataset fails."""
    pass

class TrainingDataProcessingError(Exception):
    """Raised when processing a training data row fails."""
    pass

class ResultSavingError(Exception):
    """Raised when saving processing results to a file fails."""
    pass

class JSONFileLoadError(Exception):
    """Raised when loading the ICD JSON file fails."""
    pass

class ICDCategoryCreationError(Exception):
    """Raised when creating or updating an ICDCategory fails."""
    pass

class PatientInputProcessingError(Exception):
    """Raised when processing patient input in the web interface fails."""
    pass

class ICDSearchError(Exception):
    """Raised when searching ICD codes via FTS fails."""
    pass