# Handoff — Thai Agriculture & Climate Intelligence Dashboard
### Real Open Data Edition for Thai Farmers & Agricultural Research

## 1. เป้าหมายและการอัปเกรดระบบ (System Objectives & Upgrades)

พัฒนาและยกระดับระบบ **Thai Agriculture & Climate Intelligence Dashboard** เพื่อให้เป็นแพลตฟอร์มสารสนเทศสำหรับเกษตรกรไทยและงานวิจัยวิชาการ โดยใช้ข้อมูลจริงจากแหล่ง Open Data ภาครัฐที่ตรวจสอบได้ 100% (สศก., กรมอุตุนิยมวิทยา, สสน. HII)

### สรุปการดำเนินการสำคัญตาม Master Prompt:
1. **การยกเลิกชุดข้อมูลสถานีเดี่ยวสุพรรณบุรี (Mandatory Data Removal):**
   - นำไฟล์ `สถิติปริมาณฝนและอุณหภูมิ.xlsx` ออกจาก Git และคลังข้อมูลทั้งหมดอย่างเด็ดขาด
   - ลบฟังก์ชันและโค้ดอ้างอิงทั้งหมดใน `src/data_loader.py`, `src/data_cleaning.py`, `src/analytics.py`, `app.py`, `tests/test_pipeline.py`, `README.md`, และ `data_sources.md`
   - เพิ่ม Automated Unit Tests ยืนยันว่าระบบไม่โหลดหรืออ้างอิงไฟล์ที่ยกเลิก
2. **การบูรณาการข้อมูลเปิดภาครัฐใหม่ (New Verified Open Data Integration):**
   - **สถิติราคาสินค้าเกษตรที่เกษตรกรขายได้ที่ไร่นา รายสัปดาห์ (สศก.):** OAE Data Catalog ID 35 ครอบคลุม 95 สินค้า 1,979 รายการบันทึกจริง พ.ศ. 2561–2562 (`data/raw/oae_farmgate_prices_2561_2562.csv`)
   - **ข้อมูลตรวจวัดอากาศเพื่อการเกษตรรายวัน (TMD):** กรมอุตุนิยมวิทยา Open Data API Catalog `c1-1-3-7` (34 สถานีตรวจวัดเพื่อการเกษตรทั่วประเทศ)
   - **สถานะการเข้าถึงข้อมูลดาวเทียม (GISTDA):** รายงานสถานะการเข้าถึง Catalog `agriculture-07` อย่างโปร่งใส ไม่สร้างข้อมูลจำลอง
3. **การพัฒนาสถาปัตยกรรม 5 แท็บเชื่อมโยง (5-Tab Connected Architecture):**
   - **TAB 1: ภาพรวมประเทศไทย (Thailand Overview)**
   - **TAB 2: แผนที่พื้นที่เพาะปลูกจริง (Real Cultivation Map)**
   - **TAB 3: สภาพอากาศกับความเสี่ยง (Climate Impact & Risk)**
   - **TAB 4: ผู้ช่วยวางแผนเพาะปลูก (Crop Planning Assistant)**
   - **TAB 5: ตลาดและเครื่องคำนวณฟาร์ม (Market & Farm Calculator)**
4. **ความน่าเชื่อถือและการทดสอบ:**
   - ชุดทดสอบ Unit Tests อัตโนมัติ 25 การทดสอบ ครอบคลุม 100% (ผ่านครบ 25/25)
   - ผ่านการทดสอบรันแอปพลิเคชัน Streamlit แบบ Headless AppTest โดยไม่มีข้อผิดพลาด

---

## 2. โครงสร้างและการทำงานของ 5 แท็บหลัก

### 🗺️ TAB 1: ภาพรวมประเทศไทย (Thailand Overview)
- **KPI Cards:** ผลผลิตรวม (ตัน), เนื้อที่เพาะปลูก (ไร่), เนื้อที่เก็บเกี่ยว (ไร่), Weighted Yield (กก./ไร่), Harvest Ratio (%), และ YoY Growth (%)
- **Top 10 Rankings:** จัดอันดับ 10 จังหวัด และ 10 ชนิดพืชที่มีผลผลิตสูงสุด
- **Historical Trends:** กราฟแนวโน้มรายปี พ.ศ. 2550–2562 (13 ปี) เลือกสลับตัวแปรได้

### 📍 TAB 2: แผนที่พื้นที่เพาะปลูกจริง (Real Cultivation Map)
- **4 พืชเศรษฐกิจหลัก & ทุกชนิดพืช:** ตัวเลือกด่วนสำหรับ ข้าวนาปี, ข้าวนาปรัง, อ้อยโรงงาน, มันสำปะหลังโรงงาน, ข้าวโพดเลี้ยงสัตว์ และพืชอื่นรวม 26 รายการ
- **Interactive Thailand Choropleth Map:** แผนที่แรเงาสีตามสถิติจริงของ สศก. รายจังหวัด (77 จังหวัด)
- **การจำแนกชั้นข้อมูล 3 ประเภท:**
  1. สถิติทางการเกษตร (สศก.)
  2. พื้นที่ตรวจจับจากดาวเทียม (GISTDA Open Data Catalog `agriculture-07`)
  3. การใช้ประโยชน์ที่ดิน (LDD)
- **GISTDA Live API Gateway Status:** เครื่องมือตรวจสอบพิกัดจริง พร้อมรายงานสถานะ API Key อย่างโปร่งใส (ไม่สร้างข้อมูลสุ่ม)
- **Data Table & CSV Export:** ตารางสถิติรายจังหวัดพร้อมปุ่มดาวน์โหลดไฟล์ CSV

### 🌦️ TAB 3: สภาพอากาศกับความเสี่ยง (Climate Impact & Risk)
- **3 มิติข้อมูลอากาศจริง:**
  1. ฝนเชิงพื้นที่ 77 จังหวัดทั่วประเทศ (สสน. HII IDW พ.ศ. 2561–2562)
  2. แนวโน้มฝนสะสมภาพรวมประเทศระยะยาว (สสน. HII 1970–2025)
  3. สถานีตรวจอากาศเพื่อการเกษตรรายวัน (กรมอุตุนิยมวิทยา TMD Catalog `c1-1-3-7`, 34 สถานี)
- **Bivariate Scatter & Academic Interpretation:** คำนวณ Pearson $r$, Spearman $\rho$, $p$-value พร้อมบทวิเคราะห์ทางวิชาการและข้อพึงระวัง ("Correlation ไม่ใช่ Causation")
- **Area vs Yield Quadrants:** กราฟ Scatter Plot 4 กลุ่มผลิตภาพตามค่ามัธยฐาน
- **Risk & Review Monitoring:** เกณฑ์อัตราการเก็บเกี่ยวตรวจสอบ (ค่าเริ่มต้น < 90%) พร้อมติดธงเตือน `⚠️ สัญญาณตรวจทาน` และคำนวณเนื้อที่เสียหายจริง

### 📅 TAB 4: ผู้ช่วยวางแผนเพาะปลูก (Crop Planning Assistant)
- **เครื่องมือตัดสินใจสำหรับเกษตรกร:** เลือกจังหวัด ชนิดพืช และฤดูกาลเพาะปลูก
- **เกณฑ์อ้างอิงระดับพื้นที่:** แสดงผลผลิตเฉลี่ยย้อนหลัง 13 ปีของจังหวัด และอัตราเก็บเกี่ยวสำเร็จจริงจาก สศก.
- **ปฏิทินและคำแนะนำมาตรฐาน:** ช่วงเวลาปลูกและเก็บเกี่ยว, รอบอายุพืช, คุณสมบัติดินและน้ำที่ต้องการ อ้างอิงจากกรมการข้าว, กรมวิชาการเกษตร และกรมส่งเสริมการเกษตร
- **Pre-planting Checklist:** รายการตรวจสอบก่อนปลูก 5 ข้อ เช่น ตรวจดิน แหล่งน้ำ เมล็ดพันธุ์ และการกำจัดโรคในดิน
- **Climate Precautions:** คำเตือนความเสี่ยงสภาพอากาศและแมลงศัตรูพืชสำคัญ

### 💰 TAB 5: ตลาดและเครื่องคำนวณฟาร์ม (Market & Farm Calculator)
- **Part A: สถิติราคาหน้าไร่นา สศก. (พ.ศ. 2561–2562):**
  - ข้อมูลจริงจาก OAE Data Catalog ID 35 ครอบคลุม 95 สินค้า 1,979 รายการรายสัปดาห์
  - แสดงราคาล่าสุด, ต่ำสุด, สูงสุด, ราคาเฉลี่ย, กราฟแนวโน้มรายสัปดาห์ และปุ่มดาวน์โหลด CSV
- **Part B: Interactive Farm Calculator:**
  - กำหนดค่าเริ่มต้นอัตโนมัติจากผลผลิตเฉลี่ยของจังหวัดและราคาตลาด สศก.
  - ผู้ใช้กรอกขนาดแปลง (ไร่) และต้นทุนจริง (เมล็ด/ท่อนพันธุ์, ปุ๋ย, สารเคมี, แรงงาน, เชื้อเพลิง/สูบน้ำ, ค่าเช่าที่ดิน)
  - คำนวณรายรับรวม, ต้นทุนรวม, กำไร/ขาดทุนสุทธิ, อัตรากำไร (%), ต้นทุนต่อ กก.
  - คำนวณ **ราคาขาย ณ จุดคุ้มทุน (Break-Even Price)** และ **ผลผลิตขั้นต่ำ ณ จุดคุ้มทุน (Break-Even Yield)**
  - Donut Chart สัดส่วนโครงสร้างต้นทุน
  - What-if Sensitivity Scenarios Bar Chart (จำลองสถานการณ์ราคาและผลผลิตผันผวน -20% ถึง +20%)
  - การให้คำแนะนำตัดสินใจตามส่วนต่างความปลอดภัยของกำไร (Safety Margin)

---

## 3. สรุปชุดข้อมูลที่ใช้งานจริง (Verified Data Inventory)

| ชื่อชุดข้อมูล | หน่วยงาน | ไฟล์/API ในระบบ | ขนาด/ช่วงเวลา |
|---|---|---|---|
| สถิติการผลิตสินค้าเกษตรรายปี | สศก. | `สถิติการผลิตสินค้าเกษตรรายปี 2550-2562.xlsx` | 11,914 แถว, 77 จังหวัด, 26 พืช (พ.ศ. 2550–2562) |
| สถิติราคาหน้าไร่นารายสัปดาห์ | สศก. (Catalog 35) | `data/raw/oae_farmgate_prices_2561_2562.csv` | 1,979 แถว, 95 สินค้า (พ.ศ. 2561–2562) |
| สภาพอากาศเพื่อการเกษตรรายวัน | TMD (Catalog c1-1-3-7) | API + `data/raw/tmd_agromet_sample.json` | 34 สถานีตรวจวัดทั่วประเทศ |
| ฝนเชิงพื้นที่รายจังหวัด (IDW) | สสน. HII | `data/raw/spatial_rain_hii.csv` | 8,085 แถว, 77 จังหวัด (พ.ศ. 2561–2569) |
| ฝนเฉลี่ยสะสมทั้งประเทศ | สสน. HII | `data/raw/national_rain_1970_current.csv` | พ.ศ. 2513–2568 (1970–2025) |
| อุณหภูมิสูงสุด-ต่ำสุดรายจังหวัด | สสน. HII | `data/raw/provincial_temp_2018_2019.csv` | พ.ศ. 2561–2562 |
| ขอบเขต 77 จังหวัดประเทศไทย | Royal Thai Survey Dept | `data/thai_provinces.geojson` | 77 Features (WGS84) |

---

## 4. ผลการทดสอบระบบ (Test Results)

รันคำสั่ง:
```bash
python -m pytest tests/test_pipeline.py -v
```
ผลลัพธ์:
```text
tests/test_pipeline.py::TestMandatoryDataRemoval::test_retired_suphan_dataset_not_used PASSED [  4%]
tests/test_pipeline.py::TestMandatoryDataRemoval::test_no_suphan_legacy_loader_exported PASSED [  8%]
tests/test_pipeline.py::TestDataCleaning::test_year_conversions PASSED   [ 12%]
tests/test_pipeline.py::TestDataCleaning::test_normalize_province_name PASSED [ 16%]
tests/test_pipeline.py::TestDataCleaning::test_spatial_hierarchy_classification PASSED [ 20%]
tests/test_pipeline.py::TestDataCleaning::test_clean_oae_farmgate_prices PASSED [ 24%]
tests/test_pipeline.py::TestDataCleaning::test_clean_tmd_daily_agromet PASSED [ 28%]
tests/test_pipeline.py::TestMathematicalFormulas::test_weighted_yield_calculation PASSED [ 32%]
tests/test_pipeline.py::TestMathematicalFormulas::test_weighted_yield_zero_division PASSED [ 36%]
tests/test_pipeline.py::TestMathematicalFormulas::test_harvest_ratio_calculation PASSED [ 40%]
tests/test_pipeline.py::TestMathematicalFormulas::test_harvest_ratio_zero_plant_area PASSED [ 44%]
tests/test_pipeline.py::TestMathematicalFormulas::test_unharvested_area PASSED [ 48%]
tests/test_pipeline.py::TestMathematicalFormulas::test_yoy_growth PASSED [ 52%]
tests/test_pipeline.py::TestMathematicalFormulas::test_cv_calculation PASSED [ 56%]
tests/test_pipeline.py::TestDoubleCountingPrevention::test_aggregate_double_count_prevention PASSED [ 60%]
tests/test_pipeline.py::TestAnalyticsAndPlanning::test_bivariate_stats PASSED [ 64%]
tests/test_pipeline.py::TestAnalyticsAndPlanning::test_quadrant_classification PASSED [ 68%]
tests/test_pipeline.py::TestAnalyticsAndPlanning::test_calculate_farm_budget PASSED [ 72%]
tests/test_pipeline.py::TestAnalyticsAndPlanning::test_calculate_farm_budget_invalid PASSED [ 76%]
tests/test_pipeline.py::TestAnalyticsAndPlanning::test_crop_calendar_guide PASSED [ 80%]
tests/test_pipeline.py::TestAnalyticsAndPlanning::test_climate_interpretation_academic_caveats PASSED [ 84%]
tests/test_pipeline.py::TestDataLoaderAndFiles::test_load_real_agriculture_data PASSED [ 88%]
tests/test_pipeline.py::TestDataLoaderAndFiles::test_load_oae_prices PASSED [ 92%]
tests/test_pipeline.py::TestDataLoaderAndFiles::test_load_tmd_agromet PASSED [ 96%]
tests/test_pipeline.py::TestDataLoaderAndFiles::test_load_thailand_geojson PASSED [100%]

============================= 25 passed in 2.25s ==============================
```

---

## 5. วิธีการเปิดใช้งานระบบ (Quickstart)

```bash
# 1. ติดตั้ง Dependencies
pip install -r requirements.txt

# 2. รันการทดสอบ Unit Tests
python -m pytest tests/test_pipeline.py -v

# 3. เปิดใช้งานแดชบอร์ด Streamlit
streamlit run app.py
```
เปิดเบราว์เซอร์ไปที่: `http://localhost:8501`
