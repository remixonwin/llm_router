import asyncio
import os
import sys

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from llm_router.router import IntelligentRouter


async def main():
    router = IntelligentRouter()
    await router.start()

    print(f"--- Discovered Models ({len(router.discovery.get_all_models())} total) ---")
    for m in router.discovery.get_all_models():
        print(f"{m.full_id}: {m.capabilities}")


if __name__ == "__main__":
    asyncio.run(main())
