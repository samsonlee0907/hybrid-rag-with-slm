from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AzureSettings:
    search_endpoint: str | None
    search_index: str | None
    search_api_key: str | None
    openai_endpoint: str | None
    openai_api_key: str | None
    openai_deployment: str | None
    openai_api_version: str

    @classmethod
    def from_env(cls) -> "AzureSettings":
        return cls(
            search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT") or None,
            search_index=os.getenv("AZURE_SEARCH_INDEX") or None,
            search_api_key=os.getenv("AZURE_SEARCH_API_KEY") or None,
            openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or None,
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY") or None,
            openai_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT") or None,
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )

    def validate_for_query(self) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_SEARCH_ENDPOINT": self.search_endpoint,
                "AZURE_SEARCH_INDEX": self.search_index,
                "AZURE_SEARCH_API_KEY": self.search_api_key,
                "AZURE_OPENAI_ENDPOINT": self.openai_endpoint,
                "AZURE_OPENAI_API_KEY": self.openai_api_key,
                "AZURE_OPENAI_DEPLOYMENT": self.openai_deployment,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Azure configuration: {', '.join(missing)}")

    def validate_for_search(self) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_SEARCH_ENDPOINT": self.search_endpoint,
                "AZURE_SEARCH_INDEX": self.search_index,
                "AZURE_SEARCH_API_KEY": self.search_api_key,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing Azure Search configuration: {', '.join(missing)}")
