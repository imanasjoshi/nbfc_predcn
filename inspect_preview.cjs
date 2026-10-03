// Render this project's generated notebook HTML in an isolated headless browser.
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');
(async () => {
  const browser = await chromium.launch({headless: true, channel: process.env.NBFC_BROWSER_CHANNEL || 'msedge'});
  const page = await browser.newPage({viewport: {width: 1440, height: 1080}});
  await page.goto('file:///' + path.resolve('artifacts/notebook_preview.html').replaceAll('\\', '/'));
  await page.screenshot({path: 'artifacts/notebook_opening_review.png'});
  const selectors = ['One-time untouched holdout evaluation', 'SHAP explainability', 'Model and holdout performance'];
  for (let i = 0; i < selectors.length; i++) {
    await page.getByRole('heading', {name: new RegExp(selectors[i]), exact: false}).first().scrollIntoViewIfNeeded();
    await page.screenshot({path: `artifacts/notebook_section_review_${i + 1}.png`});
  }
  const counts = await page.evaluate(() => ({
    tables: document.querySelectorAll('table').length,
    images: document.querySelectorAll('img').length,
    broken_images: [...document.images].filter(im => !im.complete || im.naturalWidth === 0).length,
    document_width: document.documentElement.scrollWidth,
    viewport_width: window.innerWidth
  }));
  fs.writeFileSync('artifacts/metrics/notebook_render_checks.json', JSON.stringify(counts, null, 2));
  console.log(JSON.stringify(counts));
  await browser.close();
})().catch(error => { console.error(error.message); process.exitCode = 1; });
