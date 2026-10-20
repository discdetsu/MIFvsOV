import { defineConfig } from '@playwright/test';

// Runs against the real stack: FastAPI (with the model) serving the production SPA build.
export default defineConfig({
	testDir: 'e2e',
	testMatch: '**/*.e2e.ts',
	use: { baseURL: 'http://localhost:8001' },
	webServer: {
		command:
			'pnpm build && cd ../backend && MIFVSOV_STATIC_DIR=../frontend/build uv run uvicorn app.main:app --port 8001',
		url: 'http://localhost:8001/api/health',
		timeout: 180_000,
		reuseExistingServer: !process.env.CI
	}
});
