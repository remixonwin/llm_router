import { defineConfig, devices } from "@playwright/test";

/**
 * Playwright configuration for testing the LLM Router in container mode.
 * This tests the standalone llmrouter service running on port 7570.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 1,
  workers: 1, // Run tests sequentially for container testing
  reporter: "list",
  use: {
    baseURL: "http://localhost:7570/",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "firefox",
      use: { ...devices["Desktop Firefox"] },
    },
  ],
  // Global timeout for all tests
  timeout: 60000,
  expect: {
    timeout: 10000,
  },
});
