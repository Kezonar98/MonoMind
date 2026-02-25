from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "MonoMind Core"
    
    # Credentials for connect
    POSTGRES_USER: str = "monomind_admin"
    POSTGRES_PASSWORD: str = "secure_dev_password_123"
    POSTGRES_DB: str = "monomind_core"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"

    @property
    def async_database_url(self) -> str:
        # PostgreSQL connection string for asyncpg
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# That string should be finded by alembic
settings = Settings()