// This acceptance regression performs the previewed cleanup through the real UI.
const assert=require('node:assert/strict');
const {chromium}=require(process.env.ASMR_TEST_PLAYWRIGHT||'playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,channel:'msedge'});
  const page=await browser.newPage();
  await page.goto(process.env.ASMR_TEST_URL||'http://127.0.0.1:7860/');
  await page.locator('#modelDevice b').first().waitFor({state:'attached'});
  const token=await page.locator('meta[name="asmr-session"]').getAttribute('content');
  const post=async(route,data)=>{
    const response=await page.request.post(new URL(`/api/${route}`,page.url()).href,{headers:{'X-ASMR-Token':token},data});
    assert(response.ok(),await response.text());return response.json();
  };
  const storage=await post('storage/get',{});
  const plan=await post('storage/scan',{root:storage.root,categories:['rebuild']});
  const bytes=plan.projects.flatMap(p=>Object.values(p.files)).reduce((sum,signature)=>sum+signature[0],0);
  assert(bytes>0,'Export a real project before running this cleanup regression');
  await page.locator('[data-page="settings"]').click();
  await page.locator('#setNav [data-set="storage"]').click();
  await page.locator('[data-clean-category="rebuild"]').click();
  await page.locator('#dialogSubmit').click();
  await page.waitForFunction(()=>!document.querySelector('#notice').hidden);
  const text=await page.locator('#notice').innerText();
  assert(text.includes(`${(bytes/1024**3).toFixed(2)} GB`),`Cleanup returned ${bytes} bytes but displayed: ${text}`);
  console.log(JSON.stringify({bytes,text}));
  await browser.close();
})().catch(error=>{console.error(error);process.exit(1)});
