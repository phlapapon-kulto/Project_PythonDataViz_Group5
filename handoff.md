# Handoff — Thai Agriculture & Climate Intelligence Dashboard

## 1. เป้าหมายและแนวทาง
สร้าง Dashboard ชื่อ **Thai Agriculture & Climate Intelligence Dashboard** เพื่อสำรวจการผลิตสินค้าเกษตรไทย ประสิทธิภาพการใช้พื้นที่ ความเสี่ยง และความสัมพันธ์กับสภาพอากาศ โดยใช้ข้อมูล Excel ที่ผู้ใช้แนบเป็นแหล่งข้อมูลหลัก พร้อมเสริม Open Data จากแหล่งทางการเมื่อจำเป็นและตรวจสอบได้จริง

Dashboard ต้องมี **3 Tabs หลัก**:
1. **ภาพรวมประเทศไทย (Thailand Overview):** ภาพรวม KPI + แผนที่ประเทศไทยแบบโต้ตอบ แสดงข้อมูลรายจังหวัด
2. **สภาพอากาศกับผลผลิต (Climate Impact):** วิเคราะห์ความสัมพันธ์ระหว่างฝน/อุณหภูมิกับ Yield หรือปริมาณผลผลิต โดยระบุข้อจำกัดด้านพื้นที่และเวลา
3. **ประสิทธิภาพและความเสี่ยง (Production & Risk):** เปรียบเทียบพืช แนวโน้ม ผลผลิตต่อไร่ ความผันผวน และสัญญาณที่ควรตรวจสอบ

คำหลัก: ข้อมูลจริง, ทุกกราฟเชื่อมกับตัวกรอง, ไม่สร้างตัวเลขปลอม, เข้าใจง่าย, แสดงที่มาและข้อจำกัดชัดเจน

## 2. ไฟล์ข้อมูลหลักที่ผู้ใช้แนบมา

### A. สถิติการผลิตสินค้าเกษตรรายปี 2550–2562
ไฟล์: `สถิติการผลิตสินค้าเกษตรรายปี 2550-2562.xlsx`
- ตรวจสอบเบื้องต้น: 11,914 แถว และ 10 คอลัมน์ (ยึดโครงสร้างจริงในไฟล์ ไม่ใช่จำนวนที่คาดเดา)
- คอลัมน์: `crop_year`, `year`, `product`, `province`, `area_plant_rai`, `area_harvest_rai`, `production_ton`, `yield_per_rai`, `harvest_ratio`, `damaged_area`
- มีหลายระดับพื้นที่ เช่น ประเทศ ภูมิภาค และจังหวัด รวมถึงหลายชนิดสินค้าเกษตร
- ปีเก็บเกี่ยวที่ตรวจพบอยู่ในช่วง พ.ศ. 2550–2562
- ก่อนรวมยอดต้องตรวจสอบแถวระดับ “ทั้งประเทศ”/ภูมิภาคที่อาจซ้ำกับผลรวมรายจังหวัด ห้ามนำระดับพื้นที่เหล่านี้มาบวกซ้ำกัน

### B. สถิติปริมาณฝนและอุณหภูมิ
ไฟล์: `สถิติปริมาณฝนและอุณหภูมิ(1).xlsx`
- ตรวจสอบเบื้องต้น: 19 แถว และ 15 คอลัมน์
- ปี พ.ศ. 2550–2568
- คอลัมน์สำคัญ: `ปี`, `ปริมาณน้ำฝน`, `จำนวนวันที่ฝนตก`, `อุณหภูมิสูงสุด`, `อุณหภูมิต่ำสุด`, `อุณหภูมิเฉลี่ยทั้งปี`, `RH Max`, `RH Min`, `RH Avg`, `หน่วยงานเจ้าของข้อมูล`
- หน่วยงานในไฟล์ระบุสถานีอุตุนิยมวิทยาจังหวัดสุพรรณบุรี
- **ข้อจำกัด:** ชุดนี้ไม่ใช่ข้อมูลสภาพอากาศทั่วประเทศ จึงห้ามนำค่าสภาพอากาศสุพรรณบุรีไปแทนจังหวัดอื่นโดยไม่แจ้งข้อจำกัด


## 2A. Required Open Data — nationwide climate coverage
Use these official Thai open-data sources to expand climate analysis beyond the supplied Suphan Buri station file. The developer must download or access the actual resource, inspect its schema, verify years, units, geography and license, and record the exact resource URL in `data_sources.md`. Dataset landing pages alone are not proof that a resource has been successfully ingested.

1. **Spatial rainfall, monthly by province — Hydro-Informatics Institute (HII)**
   - Catalog: https://data.go.th/dataset/spatial-rain
   - Resource description: monthly province-level rainfall averaged spatially from Thai Meteorological Department station observations using Inverse Distance Weighting (IDW).
   - Use: provincial rainfall choropleth, rainfall trends and matched-year analysis with agriculture.
   - Caveat: this is a spatially interpolated/derived estimate, not a direct rain-gauge reading at every location. Verify the actual available period and CSV schema.

2. **Maximum and minimum temperature, monthly by province — HII**
   - Catalog: https://data.go.th/dataset/tmax-tmin
   - Resource description: monthly provincial maximum/minimum temperature analyzed statistically from station observations.
   - Use: temperature maps and provincial climate summaries.
   - Caveat: verify coverage, temporal aggregation and whether an average temperature field is provided. Do not invent average temperature if only maximum/minimum values exist.

3. **Climate monitoring and monthly climate statistics — Thai Meteorological Department (TMD)**
   - Catalog: https://data.go.th/dataset/tmdclimatemonthly
   - Resource description: climate summaries, rainfall and maximum/minimum temperatures, monthly and weekly, based on TMD stations nationwide.
   - Use: validation/reference and supplementary climate context. Inspect the resource format and downloadable data before integrating.

4. **Supplied spreadsheet: Suphan Buri annual rainfall and temperature**
   - Keep as a local source and label it as Suphan Buri station data.
   - Use for validation against the nationwide/provincial datasets only where period, variable definitions and units are compatible.
   - Never copy these values to other provinces.

### Climate integration requirements
- Prioritize HII provincial monthly rainfall and maximum/minimum temperature as the nationwide map layer, after validating the downloadable files.
- Convert Buddhist Era (พ.ศ.) and Gregorian year consistently without losing the original year field. Normalize province names or use official province codes.
- Aggregate monthly values to annual values only using appropriate rules: sum monthly rainfall for annual rainfall totals; derive annual temperature summaries using a documented method consistent with source metadata. Never sum temperatures.
- If the source contains multiple records per province/month, investigate station/area identifiers before aggregation.
- Join climate to agriculture by province and common year only. Agricultural data in the supplied file ends in 2562; do not imply climate-production matching beyond that year unless a compatible agriculture source is added.
- Keep monthly climate data for seasonality views when crop-year/harvest-year semantics can be matched defensibly. Explicitly document whether `crop_year` or `year` is used.
- Show missing province-years as “ไม่มีข้อมูล”, not zero. Display spatial coverage, year coverage, unit, source and observation count.
- Compare Suphan Buri values in the supplied spreadsheet with compatible HII/TMD values as a validation check, but expect differences if aggregation methods or station definitions differ.
- Do not combine observed station data, spatially interpolated estimates, forecasts and climate normals as if they were the same data type.
- If downloads fail, schema is ambiguous, license is unsuitable, or coverage is insufficient, report the limitation and keep the feature clearly marked unavailable rather than fabricating data.
- Include source metadata and the exact downloaded resource links in `data_sources.md`, including access date, license/terms, date range, geographic coverage, variables, transformations and any known limitations.

## 3. ข้อกำหนดสำคัญสำหรับแผนที่ประเทศไทย
ต้องมี **แผนที่ประเทศไทยแบบ Interactive Choropleth** ใน Tab 1 โดยแสดงค่าระดับจังหวัดจากข้อมูลการผลิตสินค้าเกษตร เช่น:
- ปริมาณผลผลิตรวม (ตัน)
- พื้นที่เพาะปลูก (ไร่)
- Yield แบบถ่วงน้ำหนัก (กก./ไร่)
- Harvest Ratio (%)

### พฤติกรรมแผนที่
- ใช้ขอบเขตจังหวัดประเทศไทย (GeoJSON/TopoJSON) จากแหล่งเปิดที่เชื่อถือได้และมี license ที่เหมาะสม
- แผนที่ต้องใช้ข้อมูลจริงที่ join ด้วยชื่อหรือรหัสจังหวัดที่ normalize แล้ว
- มี dropdown ให้เลือก metric, ปี, ชนิดพืช และระดับ aggregation ตามข้อมูลที่รองรับ
- สีแผนที่ใช้ scale ต่อเนื่อง พร้อม legend และหน่วยชัดเจน
- Hover tooltip แสดงชื่อจังหวัด, metric ที่เลือก, ปี, ชนิดพืช และค่าที่คำนวณได้
- คลิกจังหวัดแล้ว cross-filter KPI, charts, ranking และตารางทั้งหน้าให้เปลี่ยนตาม
- มี search จังหวัด, ปุ่ม reset selection และตารางจังหวัดเป็นทางเลือกที่เข้าถึงได้
- ถ้าจังหวัดไม่มีข้อมูล ให้ใช้สี “ไม่มีข้อมูล” ต่างจากค่าศูนย์และอธิบายบน legend
- ไม่แสดงค่าระดับประเทศ/ภูมิภาคเป็น polygon จังหวัด และไม่บวกข้อมูลระดับประเทศซ้ำกับจังหวัด
- แสดงแหล่งที่มาของขอบเขตแผนที่และวันที่เข้าถึงใน `data_sources.md`

### ข้อจำกัดการแสดงข้อมูลอากาศบนแผนที่
- ห้ามทำแผนที่ความเสี่ยงสภาพอากาศรายจังหวัดจากไฟล์ที่มีเฉพาะสถานีสุพรรณบุรี
- ถ้าต้องการแผนที่อากาศรายจังหวัด ต้องหา Open Data ที่มีความครอบคลุมเชิงพื้นที่เหมาะสม เช่น ข้อมูลรายสถานี/รายจังหวัดจากกรมอุตุนิยมวิทยา หรือข้อมูลฝนเชิงพื้นที่จากหน่วยงานรัฐ พร้อมตรวจสอบช่วงเวลา หน่วย และ license ก่อน
- หากยังหาแหล่งข้อมูลที่เหมาะสมไม่ได้ ให้แผนที่แสดงข้อมูลการผลิตรายจังหวัด และแสดงข้อมูลอากาศเป็นกราฟของสุพรรณบุรีแยกต่างหาก พร้อมคำเตือนที่มองเห็นได้
- ห้ามสร้าง/ประมาณค่าฝนหรืออุณหภูมิของจังหวัดอื่นจากข้อมูลสุพรรณบุรี

## 4. กติกาการรวมข้อมูล
- Normalize ชื่อจังหวัดภาษาไทย (ช่องว่าง/คำขึ้นต้น/ชื่อมาตรฐาน) หรือใช้รหัสจังหวัดมาตรฐานเมื่อหาได้
- ตรวจสอบ `year` ของข้อมูลเกษตรกับ `ปี` ของข้อมูลอากาศก่อน merge
- สำหรับการวิเคราะห์อากาศกับผลผลิตในไฟล์ที่มีอยู่ ให้กรองข้อมูลเกษตรเป็นจังหวัดสุพรรณบุรีก่อน join
- ช่วงปีร่วมที่คาดว่าจะใช้ได้คือ พ.ศ. 2550–2562 แต่ต้องยืนยันจากข้อมูลที่ผ่าน cleaning จริง
- ข้อมูลอากาศปี 2563–2568 ไม่สามารถเชื่อมกับข้อมูลเกษตรชุดเดิมที่สิ้นสุดปี 2562 ได้โดยตรง เว้นแต่จะเพิ่มข้อมูลเกษตรที่ครอบคลุมปีเหล่านั้น
- ตรวจสอบ nulls, duplicates, หน่วย, ค่าติดลบ/ศูนย์ผิดปกติ และชื่อพื้นที่
- เก็บข้อมูลดิบไว้ ไม่เขียนทับไฟล์ต้นฉบับ
- แสดง coverage ปีและพื้นที่จริงของแต่ละกราฟ

## 5. ตัวชี้วัดและสูตร
- `Yield (kg/rai) = production_ton * 1000 / area_harvest_rai`
- `Harvest Ratio (%) = area_harvest_rai / area_plant_rai * 100`
- `Unharvested Area (rai) = area_plant_rai - area_harvest_rai`
- `YoY (%) = (value_this_year - value_previous_year) / value_previous_year * 100`; หากค่าปีก่อนเป็นศูนย์/ไม่มีข้อมูลให้แสดง N/A
- `CV (%) = standard_deviation / mean * 100` คำนวณจากอนุกรมรายปีที่เปรียบเทียบกันได้ พร้อมระบุจำนวนปี
- `Rainfall Anomaly = rainfall_this_year - long_term_mean`
- `Temperature Anomaly = mean_temp_this_year - long_term_mean_temp`
- Aggregated Yield ต้องคำนวณแบบถ่วงน้ำหนัก: `sum(production_ton) * 1000 / sum(area_harvest_rai)` ไม่ใช้ค่าเฉลี่ย Yield รายแถว
- Aggregated Harvest Ratio = `sum(area_harvest_rai) / sum(area_plant_rai) * 100`
- ตรวจสอบค่าที่คำนวณใหม่เทียบกับ `yield_per_rai` และ `harvest_ratio` ในไฟล์ก่อนใช้งาน

## 6. UI/UX และการทำให้เข้าใจง่าย
อ้างอิงภาพตัวอย่าง: premium modern agricultural-tech dashboard, forest green/sage/off-white, lime accent, sidebar, rounded cards, subtle borders, clean typography, spacious layout.
- ห้ามคัดลอกชื่อแบรนด์หรือโลโก้จากภาพตัวอย่าง
- Responsive desktop/tablet/mobile
- ใช้ภาษาไทยเป็นหลัก และใส่ English metric names ในวงเล็บเมื่อช่วยให้เข้าใจ
- มีคำอธิบายสั้นใต้กราฟว่า “อ่านกราฟนี้อย่างไร” หรือ insight ที่อิงข้อมูลจริง
- แสดงหน่วย ช่วงปี พื้นที่ แหล่งข้อมูล และจำนวน observation
- Global filters: ช่วงปี, ชนิดสินค้า, พื้นที่ (เมื่อมีข้อมูลรองรับ)
- แผนที่, KPI, charts, tables และ insight ต้องตอบสนองต่อ filters เดียวกัน
- มีปุ่ม reset filters และเลือก metric ของแผนที่ได้
- มี empty state ที่แนะนำวิธีแก้ เช่น “ไม่มีข้อมูลสำหรับตัวกรองนี้”
- ห้ามใส่ mock values, ข้อความ insight ที่แต่งขึ้น หรือกราฟที่ไม่เชื่อมกับข้อมูลจริง
- อย่ารวม metric คนละหน่วยบนแกนเดียวกันโดยไม่อธิบาย

## 7. Tab 1 — ภาพรวมประเทศไทย (Thailand Overview)
คำถาม: **การผลิตสินค้าเกษตรกระจายตัวอย่างไรในประเทศไทย และจังหวัดใดมีผลผลิต/ประสิทธิภาพสูง?**

องค์ประกอบ:
1. KPI cards: ผลผลิตรวม, พื้นที่เพาะปลูก, Yield แบบถ่วงน้ำหนัก, Harvest Ratio
2. แผนที่ประเทศไทยรายจังหวัด (choropleth) เปลี่ยน metric ได้ระหว่าง production, planted area, weighted yield และ harvest ratio
3. กราฟ Top 10 จังหวัดตาม metric ที่เลือก
4. กราฟ Top 10 สินค้าเกษตรตามปริมาณผลผลิต
5. Trend line ของผลผลิตรวมรายปี
6. ตารางรายละเอียดจังหวัดที่เลือก พร้อม production, area, yield, harvest ratio
7. Global filters: ปี, product, metric
8. คลิกจังหวัดบนแผนที่แล้วกราฟและ KPI ต้อง filter ตามจังหวัดนั้น
9. แสดง “ไม่มีข้อมูล” แยกจากศูนย์ และป้องกันการนับซ้ำระดับพื้นที่

## 8. Tab 2 — สภาพอากาศกับผลผลิต (Climate Impact)
คำถาม: **ฝนและอุณหภูมิสัมพันธ์กับ Yield/ผลผลิตของพืชอย่างไร?**

องค์ประกอบ:
1. KPI: ปริมาณฝน, จำนวนวันที่ฝนตก, อุณหภูมิเฉลี่ย, Yield ของพืชที่เลือก
2. Line charts แยกหน่วย: climate trends และ production/yield trends
3. Scatter plots: rainfall vs weighted yield และ mean temperature vs weighted yield
4. Correlation heatmap สำหรับ climate variables กับ Yield
5. แสดง correlation coefficient, จำนวนปีที่จับคู่ และช่วงปีที่ใช้
6. ตัวกรองชนิดพืชและช่วงปี
7. ใช้ Open Data ฝนรายจังหวัดและอุณหภูมิรายจังหวัดจาก HII เป็นแหล่งหลักสำหรับแผนที่ทั่วประเทศ เมื่อดาวน์โหลดและตรวจสอบแล้ว; ใช้ข้อมูล TMD เพื่ออ้างอิง/ตรวจสอบ และใช้ไฟล์เดิมเฉพาะสุพรรณบุรี
8. แสดงคำเตือน “ความสัมพันธ์ไม่ใช่เหตุและผล” และอธิบายข้อจำกัดจากจำนวนปี/ปัจจัยอื่น
9. ถ้าข้อมูลไม่พอ ให้แสดง empty state ไม่สร้างข้อมูลจำลอง

## 9. Tab 3 — ประสิทธิภาพและความเสี่ยง (Production & Risk)
คำถาม: **พืชชนิดใดมี Yield สูง แนวโน้มเติบโต และความผันผวนหรือสัญญาณความเสี่ยงอย่างไร?**

องค์ประกอบ:
1. KPI: production, planted area, harvested area, weighted yield, harvest ratio
2. Ranking: Top crops by production, planted area, weighted yield
3. Line chart: production/yield/harvest ratio รายปีของพืชที่เลือก
4. Heatmap: product × year สำหรับ YoY หรือ Harvest Ratio
5. Scatter plot: planted area vs weighted yield แบ่ง 4 quadrants ตาม median/percentile ที่ระบุ
6. CV ranking พร้อมจำนวนปีที่ใช้คำนวณ
7. แจ้งเตือนเมื่อ Harvest Ratio < 90% เป็น “สัญญาณที่ควรตรวจสอบ” ไม่ใช่หลักฐานยืนยันภัยธรรมชาติ
8. แสดงพื้นที่ที่ไม่ได้เก็บเกี่ยว และเปรียบเทียบปีต่อปี
9. ป้องกันการรวมระดับประเทศ/ภูมิภาค/จังหวัดซ้ำ

## 10. Open Data policy
- ใช้ Excel ที่ผู้ใช้แนบเป็น primary sources
- หากเพิ่มข้อมูล ให้พิจารณาหน่วยงานทางการ เช่น กรมอุตุนิยมวิทยา, สำนักงานเศรษฐกิจการเกษตร, data.go.th หรือชุดขอบเขตจังหวัดที่มี license ชัดเจน
- บันทึกชื่อชุดข้อมูล, หน่วยงาน, URL ต้นทาง, license/เงื่อนไข, วันที่เข้าถึง, ช่วงเวลา และ transformations ใน `data_sources.md`
- ห้ามอ้างว่าใช้ Open Data ถ้ายังไม่ได้ดาวน์โหลด/ตรวจสอบจริง
- หากไม่มีข้อมูลอากาศรายจังหวัดที่เหมาะสม ให้ระบุข้อจำกัดแทนการคาดเดา

## 11. Tech stack และโครงสร้างโค้ด
- Python 3.11+, Streamlit, Pandas, NumPy, Plotly
- GeoJSON/TopoJSON ที่มี source และ license เหมาะสมสำหรับแผนที่จังหวัด
- แยก modules: ingestion, cleaning, aggregation, metrics, mapping, charts, UI
- ใช้ relative paths/config; ไม่ hardcode absolute paths
- มี `requirements.txt`, `README.md`, `data_sources.md`, `BRD.md`, `handoff.md` และ tests
- ตรวจสอบ repository/framework เดิมก่อนแก้ ห้ามเขียนทับงานที่ไม่เกี่ยวข้อง

## 12. Testing และ Acceptance Criteria
- แอปเปิดและ run ได้ตาม README
- มี 3 Tabs ตามที่กำหนด
- แผนที่ประเทศไทยแสดงข้อมูลจังหวัดจาก dataset จริง พร้อม legend, tooltip และ no-data state
- คลิกจังหวัดแล้ว cross-filter KPI/charts/table ได้จริง
- ทุกกราฟตอบสนองต่อ filters และใช้ข้อมูลเดียวกัน
- ไม่มี mock data หรือ climate values ที่สร้างแทนจังหวัดอื่น
- ทดสอบสูตร Yield, Harvest Ratio, YoY, CV และการ merge ปี/จังหวัด
- ทดสอบการป้องกัน double counting ระหว่าง national/region/province
- แสดง source, unit, period, geography, observation count และ caveats
- รัน tests และตรวจสอบว่าแอปเริ่มทำงานได้ก่อนส่งมอบ

## 13. GitHub / Run / Deployment
1. ตรวจสอบ repository, branch, git status และ remote ก่อน
2. ห้ามลบงานเดิมหรือเปลี่ยน visibility
3. ห้าม commit secrets, tokens, credentials หรือ environment files
4. รัน tests และเปิดแอปจริงใน local environment
5. commit ด้วยข้อความสื่อความหมาย และ push ไป branch ที่เหมาะสม
6. ห้าม force push
7. Deploy เฉพาะเมื่อมี integration/สิทธิ์ที่ได้รับอนุญาต; หากต้องล็อกอินหรืออนุมัติ ให้แจ้งผู้ใช้
8. รายงาน repo URL, branch, commit hash, test result, run command และ deployment URL ที่ตรวจสอบแล้ว
9. ห้ามอ้างว่า push/deploy สำเร็จหากยังไม่เกิดขึ้นจริง

## 14. Deliverables
- Dashboard source code
- `handoff.md`
- `BRD.md`
- `README.md`
- `data_sources.md`
- `requirements.txt`
- Automated tests
- แผนที่ประเทศไทยแบบ interactive
- 3 Tabs ที่เชื่อมโยงกันและใช้ข้อมูลจริง
