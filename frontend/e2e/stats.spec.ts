import { test, expect } from "@playwright/test";

test.describe("Stats Page", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/llm/stats");
  });

  test("should load stats page with heading", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Statistics" })).toBeVisible();
  });

  test("should display stats cards after loading", async ({ page }) => {
    await expect(page.getByText("Total Models")).toBeVisible({ timeout: 10000 });
    await expect(page.getByText("Cache Hits")).toBeVisible();
    await expect(page.getByText("Cache Misses")).toBeVisible();
  });

  test("should display provider quotas section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Provider Quotas" })).toBeVisible({
      timeout: 10000,
    });
  });

  test("should navigate to Admin page", async ({ page }) => {
    await page.getByRole("link", { name: "Admin" }).click();
    await expect(page).toHaveURL("/llm/admin");
    await expect(page.getByRole("heading", { name: "Admin" })).toBeVisible();
  });
});
