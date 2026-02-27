import { test, expect } from "@playwright/test";

/**
 * API Integration Tests - Container Version
 * Tests that the frontend can communicate with the backend API
 */
test.describe("API Integration (Container)", () => {
  test("should fetch health endpoint", async ({ page }) => {
    // Intercept the health API call
    const responsePromise = page.waitForResponse(response => 
      response.url().includes("/health")
    );
    
    await page.goto("/");
    
    const response = await responsePromise;
    expect(response.status()).toBe(200);
    
    const data = await response.json();
    expect(data).toHaveProperty("status");
    expect(data).toHaveProperty("providers_available");
  });

  test("should fetch providers endpoint", async ({ page }) => {
    const response = await page.request.get("/providers");
    expect(response.ok()).toBe(true);
    
    const data = await response.json();
    expect(data).toHaveProperty("providers");
  });

  test("should fetch models endpoint", async ({ page }) => {
    const response = await page.request.get("/v1/models");
    expect(response.ok()).toBe(true);
    
    const data = await response.json();
    expect(data).toHaveProperty("data");
    expect(Array.isArray(data.data)).toBe(true);
  });

  test("should fetch stats endpoint", async ({ page }) => {
    const response = await page.request.get("/stats");
    expect(response.ok()).toBe(true);
    
    const data = await response.json();
    expect(data).toHaveProperty("providers");
  });

  test("should handle CORS preflight", async ({ request }) => {
    const response = await request.fetch("/health", {
      method: "OPTIONS",
      headers: {
        "Origin": "http://localhost:7570",
        "Access-Control-Request-Method": "GET",
      },
    });
    
    // The server should at least not error on OPTIONS
    expect([200, 204, 404].includes(response.status())).toBe(true);
  });
});
