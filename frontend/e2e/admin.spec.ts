import { test, expect } from "@playwright/test";

test.describe("Admin Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/admin");
  });

  test("should load admin page with heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Admin" })).toBeVisible();
  });

  test("should display cache management section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Cache Management" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Clear Cache/i })).toBeVisible();
  });

  test("should display quota management section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Quota Management" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Reset Quotas/i })).toBeVisible();
  });

  test("should display model discovery section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Model Discovery" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Refresh Models/i })).toBeVisible();
  });

  test("should navigate to Dashboard", async ({ page }) => {
    await page.getByRole("link", { name: "Dashboard" }).click();
    await expect(page).toHaveURL("/");
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();
  });

  test("should display OpenAI-compatible section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "OpenAI Compatible Endpoints" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Add Endpoint/i })).toBeVisible();
  });

  test("should add new OpenAI-compatible endpoint", async ({ page }) => {
    await page.getByRole("button", { name: /Add Endpoint/i }).click();

    await page.getByLabel("Base URL").fill("http://host.docker.internal:7330/v1");
    await page.getByLabel("Default Models").fill("llama-3.1-8b");
    await page.getByRole("button", { name: "Add Endpoint" }).click();

    await expect(page.getByText("http://host.docker.internal:7330/v1")).toBeVisible();
  });

  test("should test OpenAI-compatible endpoint connection", async ({ page }) => {
    await expect(page.getByText("http://host.docker.internal:7330/v1")).toBeVisible();

    await page.getByRole("button", { name: /Test/i }).first().click();

    await expect(page.getByText(/success|connected/i, { exact: false })).toBeVisible({
      timeout: 10000,
    });
  });
});
