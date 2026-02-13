from multiprocessing import cpu_count
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource, PydanticBaseSettingsSource

class DataSettings(BaseSettings):
    data_dir: str = "data"
    cache_enabled: bool = True
    parallel_workers: int = 0

    def get_workers(self):
        return self.parallel_workers or max(1, cpu_count() - 1)

class SourceSettings(BaseSettings):
    house_metadata_url: str = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.ZIP"
    house_pdf_url: str = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"

class AnalysisSettings(BaseSettings):
    default_year: int = 2025
    default_horizons: list[int] = Field(default_factory=lambda: [90])
    default_threshold: float = 5.0

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file='config.toml',
        env_nested_delimiter='__',
        extra='ignore'
    )

    data: DataSettings = Field(default_factory=DataSettings)
    sources: SourceSettings = Field(default_factory=SourceSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings, TomlConfigSettingsSource(settings_cls), env_settings, file_secret_settings)
