import { test, expect } from "@playwright/test";

/**
 * Stats Page Tests - Container Version
 */
test.describe("Stats Page (Container)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/stats");
  });

  test("should load stats page with heading", async ({ page }) => {
    await expect(page.locator("h2").filter({ hasText: "Statistics" })).toBeVisible();
  });

  test("should display statistics content", async ({ page }) => {
    // Check for common stats sections
    await expect(page.getByText(/providers|models|cache|requests/i).first()).toBeVisible({
      timeout: 10000,
    });
  });

  test("should navigate to Dashboard", async ({ page }) => {
    await page.getByRole("link", { name: "Dashboard" }).click();
    await expect(page).toHaveURL("/");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
  });

  test("should navigate to Models", async ({ page }) => {
    await page.getByRole("link", { name: "Models" }).click();
    await expect(page).toHaveURL("/models");
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  });

  test("should navigate to Admin", async ({ page }) => {
    await page.getByRole("link", { name: "Admin" }).click();
    await expect(page).toHaveURL("/admin");
    await expect(page.getByRole("heading", { name: "Admin" })).toBeVisible();
  });

  test("should handle page refresh", async ({ page }) => {
    await page.reload();
    await expect(page.getByRole("heading", { name: "Statistics" })).toBeVisible({ timeout: 10000 });
  });
});
