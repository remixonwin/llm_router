import { test, expect } from "@playwright/test";

test.describe("API Keys Management", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/llm/admin");
  });

  test("should display API keys section", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "API Keys" })).toBeVisible();
    await expect(page.getByText("Configure API keys for each provider")).toBeVisible();
  });

  test("should display configured and unconfigured providers", async ({ page }) => {
    await expect(page.getByText("/ 13 configured")).toBeVisible({ timeout: 10000 });
  });

  test("should have input field for entering API key", async ({ page }) => {
    const inputs = page.getByPlaceholder(/Enter API key/);
    await expect(inputs.first()).toBeVisible();
  });

  test("should have show/hide toggle for API key input", async ({ page }) => {
    await expect(
      page
        .getByRole("button")
        .filter({ has: page.locator("svg") })
        .first()
    ).toBeVisible();
  });
});
