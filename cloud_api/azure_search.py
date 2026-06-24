from __future__ import annotations

import httpx

from cloud_api.settings import AzureSettings


class AzureHybridRetriever:
    def __init__(self, settings: AzureSettings) -> None:
        settings.validate_for_query()
        self.settings = settings

    async def retrieve(self, query: str, *, site_id: str, top_k: int = 5) -> list[dict]:
        """Run Azure AI Search hybrid retrieval.

        This uses keyword search plus filters as a conservative baseline. Add
        vectorQueries once the cloud embedding/vectorizer path is configured.
        """
        assert self.settings.search_endpoint
        assert self.settings.search_index
        assert self.settings.search_api_key

        url = (
            f"{self.settings.search_endpoint.rstrip('/')}/indexes/"
            f"{self.settings.search_index}/docs/search?api-version=2024-07-01"
        )
        payload = {
            "search": query,
            "filter": f"site_id eq '{site_id}'",
            "top": top_k,
            "select": "case_id,title,problem,resolution,source_uri,risk_level",
        }
        headers = {"api-key": self.settings.search_api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        data = response.json()
        return [
            {
                "case_id": item.get("case_id"),
                "title": item.get("title"),
                "snippet": item.get("resolution") or item.get("problem"),
                "source_uri": item.get("source_uri"),
                "risk_level": item.get("risk_level"),
                "score": item.get("@search.score", 0),
            }
            for item in data.get("value", [])
        ]

    async def grounded_chat(self, query: str, hits: list[dict]) -> str:
        assert self.settings.openai_endpoint
        assert self.settings.openai_api_key
        assert self.settings.openai_deployment

        evidence = "\n".join(
            f"[C{i}] {hit.get('case_id')} {hit.get('title')} {hit.get('snippet')} {hit.get('source_uri')}"
            for i, hit in enumerate(hits, start=1)
        )
        url = (
            f"{self.settings.openai_endpoint.rstrip('/')}/openai/deployments/"
            f"{self.settings.openai_deployment}/chat/completions"
            f"?api-version={self.settings.openai_api_version}"
        )
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a construction field copilot. Answer only from cited case evidence and escalate high-risk work.",
                },
                {"role": "user", "content": f"EVIDENCE:\n{evidence}\n\nQUESTION:\n{query}"},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        headers = {"api-key": self.settings.openai_api_key, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
