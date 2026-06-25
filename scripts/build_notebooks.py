from __future__ import annotations

from build_final_demo_notebooks import main as build_final_demo_notebooks


if __name__ == "__main__":
    print("scripts/build_notebooks.py is deprecated; building the canonical Local-Offline-RAG, Local-Offline-RAG-tc, and Hybrid-RAG notebooks.")
    build_final_demo_notebooks()
