import { test, expect } from "@playwright/test";

/**
 * Admin Page Tests - Container Version
 */
test.describe("Admin Page (Container)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/admin");
  });

  test("should load admin page with heading", async ({ page }) => {
    await expect(page.locator("h2").filter({ hasText: "Admin" })).toBeVisible();
  });

  test("should display cache management section", async ({ page }) => {
    await expect(page.getByText("Cache Management")).toBeVisible();
    await expect(page.getByRole("button", { name: /Clear Cache/i })).toBeVisible();
  });

  test("should display quota management section", async ({ page }) => {
    await expect(page.getByText("Quota Management")).toBeVisible();
    await expect(page.getByRole("button", { name: /Reset Quotas/i })).toBeVisible();
  });

  test("should display model discovery section", async ({ page }) => {
    await expect(page.getByText("Model Discovery")).toBeVisible();
    await expect(page.getByRole("button", { name: /Refresh Models/i })).toBeVisible();
  });

  test("should display OpenAI-compatible section", async ({ page }) => {
    await expect(page.getByText("OpenAI Compatible Endpoints")).toBeVisible();
    await expect(page.getByRole("button", { name: /Add Endpoint/i })).toBeVisible();
  });

  test("should open add endpoint dialog", async ({ page }) => {
    await page.getByRole("button", { name: /Add Endpoint/i }).click();
    
    // Check that the dialog/form appears
    await expect(page.getByPlaceholder("https://api.example.com/v1")).toBeVisible();
    await expect(page.getByPlaceholder("model1,model2")).toBeVisible();
  });

  test("should navigate to Dashboard", async ({ page }) => {
    await page.getByRole("link", { name: "Dashboard" }).click();
    await expect(page).toHaveURL("/");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
  });

  test("should navigate to Stats", async ({ page }) => {
    await page.getByRole("link", { name: "Stats" }).click();
    await expect(page).toHaveURL("/stats");
    await expect(page.getByRole("heading", { name: "Statistics" })).toBeVisible();
  });

  test("should handle page refresh", async ({ page }) => {
    await page.reload();
    await expect(page.getByRole("heading", { name: "Admin" })).toBeVisible({ timeout: 10000 });
  });

  test("should display API keys section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: /API Keys|Provider Keys/i })).toBeVisible();
  });
});
