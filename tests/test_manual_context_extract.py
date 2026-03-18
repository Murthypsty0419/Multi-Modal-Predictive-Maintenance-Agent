import asyncio
import json
from app.core.manual_context import manual_context_node

# Example test state (customize anomaly_query as needed)
test_state = {
    "anomaly_query": "maximum pressure and warranty information for pump"
}

async def main():
    result = await manual_context_node(test_state)
    # Save output as JSON
    with open("tests/manual_context_test_output.json", "w") as f:
        json.dump(result["manual_context"], f, indent=2)
    print("Manual context extraction complete. Output saved to tests/manual_context_test_output.json")

if __name__ == "__main__":
    asyncio.run(main())
