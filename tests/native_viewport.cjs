// Run against the portable interface with ASMR_TEST_PLAYWRIGHT pointing to Playwright.
const assert=require('node:assert/strict');
const {chromium}=require(process.env.ASMR_TEST_PLAYWRIGHT||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  const page=await browser.newPage();
  await page.goto(process.env.ASMR_TEST_URL||'http://127.0.0.1:7860/');
  await page.locator('#modelDevice b').first().waitFor({state:'attached'});
  for(const width of [1440,900,390]){
    await page.setViewportSize({width,height:1000});
    for(const name of ['home','models','batch','settings']){
      await page.locator(`[data-page="${name}"]`).click();
      if(name==='settings')await page.locator('#setNav [data-set="general"]').click();
      const actual=await page.evaluate(()=>document.documentElement.scrollWidth);
      assert(actual<=width,`${name} at ${width}px overflows to ${actual}px`);
      if(name==='settings')for(const tab of ['keys','d-asr','d-tr','d-tts','d-mix','batch','storage','logs']){
        await page.locator(`#setNav [data-set="${tab}"]`).click();
        const actual=await page.evaluate(()=>document.documentElement.scrollWidth);
        assert(actual<=width,`settings/${tab} at ${width}px overflows to ${actual}px`);
      }
    }
  }
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
