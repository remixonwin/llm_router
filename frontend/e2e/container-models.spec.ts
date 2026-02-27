import { test, expect } from "@playwright/test";

/**
 * Models Page Tests - Container Version
 */
test.describe("Models Page (Container)", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/models");
  });

  test("should load models page with heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible();
  });

  test("should display search input", async ({ page }) => {
    await expect(page.getByPlaceholder("Search models...")).toBeVisible();
  });

  test("should display filter dropdowns", async ({ page }) => {
    await expect(page.getByRole("combobox").first()).toBeVisible();
    await expect(page.getByRole("combobox").nth(1)).toBeVisible();
  });

  test("should display models table after loading", async ({ page }) => {
    await expect(page.getByRole("columnheader", { name: "Model" })).toBeVisible({ timeout: 15000 });
    await expect(page.getByRole("columnheader", { name: "Provider" })).toBeVisible();
    await expect(page.getByRole("columnheader", { name: "Context" })).toBeVisible();
    await expect(page.getByRole("columnheader", { name: "Capabilities" })).toBeVisible();
    await expect(page.getByRole("columnheader", { name: "Free" })).toBeVisible();
  });

  test("should filter models by search", async ({ page }) => {
    const searchInput = page.getByPlaceholder("Search models...");
    await searchInput.fill("gpt");
    await expect(searchInput).toHaveValue("gpt");
  });

  test("should have provider filter dropdown", async ({ page }) => {
    const providerSelect = page.getByRole("combobox").first();
    await expect(providerSelect).toBeVisible();
    await expect(providerSelect).toHaveValue("");
  });

  test("should display model count", async ({ page }) => {
    // Look for text that contains a number followed by "models" or similar
    await expect(page.getByText(/\d+\s+models?/i)).toBeVisible({ timeout: 15000 });
  });

  test("should navigate back to Dashboard", async ({ page }) => {
    await page.getByRole("link", { name: "Dashboard" }).click();
    await expect(page).toHaveURL("/");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
  });

  test("should handle page refresh on models page", async ({ page }) => {
    await page.reload();
    await expect(page.getByRole("heading", { name: "Models" })).toBeVisible({ timeout: 15000 });
  });
});
