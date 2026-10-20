"""Print the OpenAPI schema (used by the frontend's `pnpm gen:api`)."""

import json

from app.main import app

if __name__ == "__main__":
    print(json.dumps(app.openapi(), indent=2))
