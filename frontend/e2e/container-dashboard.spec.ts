import { test, expect } from "@playwright/test";

/**
 * Dashboard Page Tests - Container Version
 * Tests the LLM Router frontend running in standalone container mode on port 7570
 */
test.describe("Dashboard Page (Container)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("should load dashboard and display header", async ({ page }) => {
    await expect(page.locator("h1")).toContainText("LLM Router");
  });

  test("should display navigation links", async ({ page }) => {
    await expect(page.getByRole("link", { name: "Dashboard" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Models" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Stats" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Admin" })).toBeVisible();
  });

  test("should display dashboard content after loading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
    await expect(page.getByText("Status")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText("Providers Available")).toBeVisible();
    await expect(page.getByText("Total Providers")).toBeVisible();
  });

  test("should display provider health section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Provider Health" })).toBeVisible({
      timeout: 10000,
    });
  });

  test("should display health status correctly", async ({ page }) => {
    // Check that status is shown (healthy/degraded)
    const statusElement = page.locator("text=/healthy|degraded|error/i").first();
    await expect(statusElement).toBeVisible({ timeout: 10000 });
  });

  test("should navigate to Models page", async ({ page }) => {
    await page.getByRole("link", { name: "Models" }).click();
    await expect(page).toHaveURL("/models");
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  });

  test("should navigate to Stats page", async ({ page }) => {
    await page.getByRole("link", { name: "Stats" }).click();
    await expect(page).toHaveURL("/stats");
    await expect(page.getByRole("heading", { name: "Statistics" })).toBeVisible();
  });

  test("should navigate to Admin page", async ({ page }) => {
    await page.getByRole("link", { name: "Admin" }).click();
    await expect(page).toHaveURL("/admin");
    await expect(page.getByRole("heading", { name: "Admin" })).toBeVisible();
  });

  test("should handle page refresh on dashboard", async ({ page }) => {
    await page.reload();
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible({ timeout: 10000 });
    await expect(page.locator("h1")).toContainText("LLM Router");
  });
});
