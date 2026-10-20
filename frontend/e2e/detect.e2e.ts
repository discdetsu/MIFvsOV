import { expect, test } from '@playwright/test';
import path from 'node:path';

const SAMPLE = path.resolve(
	import.meta.dirname,
	'../../backend/tests/fixtures/images/Opisthorchis viverrine_1007.jpg'
);

test('upload an image and see detection boxes', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('heading', { name: 'MIF & OV Egg Detection' })).toBeVisible();

	await page.getByTestId('file-input').first().setInputFiles(SAMPLE);

	const boxes = page.getByTestId('overlay').locator('rect');
	await expect(boxes.first()).toBeVisible({ timeout: 30_000 });
	expect(await boxes.count()).toBeGreaterThanOrEqual(1);
	await expect(page.getByTestId('count-MIF')).toHaveText('1');

	const [download] = await Promise.all([
		page.waitForEvent('download'),
		page.getByRole('button', { name: 'Download annotated PNG' }).click()
	]);
	expect(download.suggestedFilename()).toBe('Opisthorchis viverrine_1007-annotated.png');

	// Hiding MIF removes its box from the overlay.
	await page.getByRole('button', { name: 'Show MIF' }).click();
	await expect(page.getByTestId('count-MIF')).toHaveText('0');
});
