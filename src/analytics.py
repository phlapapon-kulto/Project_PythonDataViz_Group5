"""
Analytics module for Thai Agriculture & Climate Intelligence Dashboard.
Implements correlation analysis, statistical tests, quadrant classification,
ranking algorithms, and automated data-driven interpretation generation.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


def calculate_farm_budget(
    planted_area_rai: float,
    expected_yield_kg_rai: float,
    selling_price_baht_kg: float,
    seeds_cost: float = 0.0,
    fertilizer_cost: float = 0.0,
    water_energy_cost: float = 0.0,
    labor_cost: float = 0.0,
    land_rent_cost: float = 0.0,
    other_cost: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate farm budget, cost per unit, break-even price/yield, and sensitivity scenarios.
    For Tab 5: Market & Farm Calculator.
    """
    if planted_area_rai <= 0 or expected_yield_kg_rai <= 0 or selling_price_baht_kg <= 0:
        return {
            "is_valid": False,
            "message": "กรุณากรอกเนื้อที่เพาะปลูก ผลผลิตคาดการณ์ และราคาขายให้มากกว่า 0"
        }

    total_yield_kg = planted_area_rai * expected_yield_kg_rai
    total_yield_ton = total_yield_kg / 1000.0

    gross_revenue = total_yield_kg * selling_price_baht_kg
    total_cost = (
        seeds_cost + fertilizer_cost + water_energy_cost +
        labor_cost + land_rent_cost + other_cost
    )

    cost_per_rai = total_cost / planted_area_rai
    cost_per_kg = total_cost / total_yield_kg if total_yield_kg > 0 else np.nan

    net_profit = gross_revenue - total_cost
    profit_margin_pct = (net_profit / gross_revenue * 100.0) if gross_revenue > 0 else np.nan

    # Break-even calculations
    # Break-even price: Price where revenue = cost -> total_cost / total_yield_kg
    breakeven_price_kg = total_cost / total_yield_kg if total_yield_kg > 0 else np.nan
    # Break-even yield: Yield where revenue = cost -> total_cost / (selling_price * planted_area)
    breakeven_yield_kg_rai = total_cost / (selling_price_baht_kg * planted_area_rai) if selling_price_baht_kg > 0 else np.nan

    # What-if sensitivity scenarios
    scenarios = []
    for price_factor, p_desc in [(1.1, "+10%"), (1.0, "ฐาน (Base)"), (0.9, "-10%"), (0.8, "-20%")]:
        for yield_factor, y_desc in [(1.0, "ฐาน (Base)"), (0.9, "-10%"), (0.8, "-20%")]:
            sim_rev = (total_yield_kg * yield_factor) * (selling_price_baht_kg * price_factor)
            sim_profit = sim_rev - total_cost
            scenarios.append({
                "price_change": p_desc,
                "yield_change": y_desc,
                "revenue": sim_rev,
                "profit": sim_profit,
                "is_profitable": sim_profit >= 0
            })

    return {
        "is_valid": True,
        "planted_area_rai": planted_area_rai,
        "expected_yield_kg_rai": expected_yield_kg_rai,
        "total_yield_kg": total_yield_kg,
        "total_yield_ton": total_yield_ton,
        "selling_price_baht_kg": selling_price_baht_kg,
        "gross_revenue": gross_revenue,
        "total_cost": total_cost,
        "cost_per_rai": cost_per_rai,
        "cost_per_kg": cost_per_kg,
        "net_profit": net_profit,
        "profit_margin_pct": profit_margin_pct,
        "breakeven_price_kg": breakeven_price_kg,
        "breakeven_yield_kg_rai": breakeven_yield_kg_rai,
        "scenarios": scenarios
    }


def get_crop_calendar_guide(product: str) -> Dict[str, Any]:
    """
    Retrieve official crop calendar, seasonal guidelines, and pre-planting checklists
    from the Rice Department / Department of Agriculture / Department of Agricultural Extension.
    """
    guides = {
        "ข้าวนาปี": {
            "crop_group": "พืชไร่ / ข้าว",
            "planting_window": "พฤษภาคม – กรกฎาคม (ช่วงต้นถึงกลางฤดูฝน)",
            "harvest_window": "ตุลาคม – ธันวาคม",
            "growth_duration_days": "120 – 150 วัน (ขึ้นกับพันธุ์ไวแสงหรือไม่ไวแสง)",
            "water_soil_req": "ดินเหนียวหรือดินร่วนปนเหนียว เก็บกักน้ำได้ดี ปริมาณน้ำฝนสะสมตลอดฤดู 1,200–1,500 มม. ค่า pH 5.5–6.5",
            "pre_planting_checklist": [
                "ตรวจสอบแหล่งน้ำและสภาพดิน (ไถดะตากดิน 7–10 วันเพื่อกำจัดวัชพืชและโรค)",
                "คัดเลือกเมล็ดพันธุ์บริสุทธิ์จากศูนย์เมล็ดพันธุ์ข้าว (อัตรา 15–20 กก./ไร่ สำหรับหว่าน)",
                "แช่เมล็ดพันธุ์ด้วยน้ำสะอาด 1 คืน ผึ่งให้งอกตุ่มตา 1 คืนก่อนหว่าน",
                "ปรับระดับดินแปลงนาให้สม่ำเสมอเพื่อประสิทธิภาพการคุมระดับน้ำ",
                "เตรียมปุ๋ยรองพื้นตามค่าวิเคราะห์ดิน (สูตร 16-20-0 หรือ 16-16-8)"
            ],
            "key_cautions": "เฝ้าระวังช่วงฝนทิ้งช่วงในเดือนมิถุนายน–กรกฎาคม และโรคไหม้ข้าว เพลี้ยกระโดดสีน้ำตาลในช่วงที่มีความชื้นสูง",
            "source_agency": "กรมการข้าว กระทรวงเกษตรและสหกรณ์"
        },
        "ข้าวนาปรัง": {
            "crop_group": "พืชไร่ / ข้าว",
            "planting_window": "พฤศจิกายน – มกราคม (หลังเก็บเกี่ยวข้าวนาปี)",
            "harvest_window": "กุมภาพันธ์ – เมษายน",
            "growth_duration_days": "100 – 120 วัน (พันธุ์ไม่ไวต่อช่วงแสง เช่น กข41, กข47, พิษณุโลก 2)",
            "water_soil_req": "ต้องอยู่ในเขตชลประทานหรือมีแหล่งน้ำสำรองเพียงพอตลอดฤดูปลูก ใช้น้ำเฉลี่ย 1,000–1,200 ลบ.ม./ไร่",
            "pre_planting_checklist": [
                "ตรวจสอบปริมาณน้ำในอ่างเก็บน้ำหรือคลองส่งน้ำชลประทานก่อนตัดสินใจปลูก",
                "ใช้พันธุ์ข้าวไม่ไวต่อช่วงแสงที่มีอายุสั้นและทนต่อสภาพอากาศเย็นช่วงแรก",
                "ไถดะและไถแปรปรับเทือกให้ราบเรียบเพื่อประหยัดน้ำชลประทาน",
                "วางแผนการให้น้ำแบบเปียกสลับแห้งแกล้งข้าวเพื่อลดต้นทุนค่าน้ำมันและลดการเกิดก๊าซมีเทน"
            ],
            "key_cautions": "ห้ามปลูกนอกเขตชลประทานที่ไม่มีแหล่งน้ำยืนยัน ระวังอุณหภูมิต่ำกระทบต่อการงอกในภาคเหนือ/อีสาน",
            "source_agency": "กรมการข้าว และกรมชลประทาน"
        },
        "ข้าวโพดเลี้ยงสัตว์": {
            "crop_group": "พืชไร่ / ข้าวโพด",
            "planting_window": "รุ่นที่ 1: พฤษภาคม – มิถุนายน (ฤดูฝน) | รุ่นที่ 2: พฤศจิกายน – ธันวาคม (หลังนา)",
            "harvest_window": "สิงหาคม – กันยายน (รุ่น 1) | กุมภาพันธ์ – มีนาคม (รุ่น 2)",
            "growth_duration_days": "105 – 120 วัน",
            "water_soil_req": "ดินร่วน ดินร่วนเหนียว ระบายน้ำดี ไม่ชอบน้ำขังขังแฉะ ต้องการฝน 500–800 มม. ค่า pH 5.5–7.0",
            "pre_planting_checklist": [
                "ยกร่องแปลงปลูกเพื่อป้องกันน้ำขังโคนต้นและรากเน่า",
                "ใช้เมล็ดพันธุ์ลูกผสมเดี่ยวที่ทนทานต่อโรคราน้ำค้างและหนอนเจาะสมอฝ้าย",
                "คลุกเมล็ดพันธุ์ด้วยสารป้องกันกำจัดเชื้อราและแมลงก่อนหยอด",
                "ตรวจวัดความชื้นในดินก่อนหยอดเมล็ดให้ลึก 3–5 ซม."
            ],
            "key_cautions": "ระวังหนอนกระทู้ข้าวโพดลายจุด (Fall Armyworm) ทำลายยอด และไม่ควรปลูกในพื้นที่ลุ่มน้ำท่วมขัง",
            "source_agency": "กรมวิชาการเกษตร และกรมส่งเสริมการเกษตร"
        },
        "อ้อยโรงงาน": {
            "crop_group": "พืชไร่ / อ้อย",
            "planting_window": "อ้อยต้นฝน: เมษายน – พฤษภาคม | อ้อยปลายฝน: ตุลาคม – ธันวาคม",
            "harvest_window": "ธันวาคม – เมษายน (ตามฤดูเปิดหีบของโรงงานน้ำตาล)",
            "growth_duration_days": "10 – 14 เดือน (300 – 420 วัน)",
            "water_soil_req": "ดินร่วน ดินร่วนปนทราย หรือดินร่วนเหนียว ระบายน้ำดี ไม่ชอบน้ำขัง ต้องการน้ำ 1,200–1,500 มม./ปี ค่า pH 5.5–7.0",
            "pre_planting_checklist": [
                "ไถดะ ไถแปร และไถระเบิดดินดานลึก 40–50 ซม. เพื่อให้รากอ้อยหยั่งลึกทนแล้ง",
                "คัดเลือกท่อนพันธุ์อ้อยอายุ 8–10 เดือน มีตาสมบูรณ์ ปราศจากโรคขาวใบและหนอนกอ",
                "แช่ท่อนพันธุ์ในสารป้องกันกำจัดเชื้อราและแมลงก่อนปลูก",
                "ยกร่องลึก 25–30 ซม. ระยะระหว่างร่อง 1.3–1.5 เมตร เพื่อรองรับรถตัดอ้อย",
                "ใส่ปุ๋ยรองพื้นสูตร 16-16-8 หรือ 15-15-15 อัตรา 30–50 กก./ไร่"
            ],
            "key_cautions": "ระวังหนอนกออ้อยระบาดช่วงแล้ง และงดการเผาอ้อยก่อนเก็บเกี่ยวเพื่อรักษาคุณภาพดินและลดฝุ่น PM2.5",
            "source_agency": "สำนักงานคณะกรรมการอ้อยและน้ำตาลทราย และกรมวิชาการเกษตร"
        },
        "มันสำปะหลังโรงงาน": {
            "crop_group": "พืชไร่ / มันสำปะหลัง",
            "planting_window": "ต้นฝน: เมษายน – พฤษภาคม | ปลายฝน: ตุลาคม – พฤศจิกายน",
            "harvest_window": "8 – 12 เดือนหลังปลูก (มกราคม – เมษายน)",
            "growth_duration_days": "240 – 360 วัน",
            "water_soil_req": "ดินร่วนปนทรายหรือดินร่วนลูกรัง ระบายน้ำดี ทนแล้งสูง ต้องการปริมาณฝน 800–1,200 มม. ค่า pH 5.0–6.5",
            "pre_planting_checklist": [
                "คัดเลือกท่อนพันธุ์สมบูรณ์ อายุ 8–12 เดือน ปราศจากโรคใบด่างมันสำปะหลัง",
                "แช่ท่อนพันธุ์ด้วยสารป้องกันกำจัดแมลงหวี่ขาวยาสูบและเพลี้ยแป้ง 5–10 นาที",
                "ยกร่องสูง 30–40 ซม. เพื่อเพิ่มความโปร่งของดินให้หัวขยายตัวได้ดี",
                "ปักท่อนพันธุ์ในแนวดิ่งหรือเอียงเล็กน้อย ลึก 5–10 ซม."
            ],
            "key_cautions": "ตรวจแปลงสม่ำเสมอเพื่อเฝ้าระวังโรคใบด่างมันสำปะหลัง (Cassava Mosaic Disease) ที่ถ่ายทอดทางแมลงหวี่ขาว",
            "source_agency": "กรมวิชาการเกษตร"
        },
        "มันสำปะหลัง": {
            "crop_group": "พืชไร่ / มันสำปะหลัง",
            "planting_window": "ต้นฝน: เมษายน – พฤษภาคม | ปลายฝน: ตุลาคม – พฤศจิกายน",
            "harvest_window": "8 – 12 เดือนหลังปลูก (มกราคม – เมษายน)",
            "growth_duration_days": "240 – 360 วัน",
            "water_soil_req": "ดินร่วนปนทรายหรือดินร่วนลูกรัง ระบายน้ำดี ทนแล้งสูง ต้องการปริมาณฝน 800–1,200 มม. ค่า pH 5.0–6.5",
            "pre_planting_checklist": [
                "คัดเลือกท่อนพันธุ์สมบูรณ์ อายุ 8–12 เดือน ปราศจากโรคใบด่างมันสำปะหลัง",
                "แช่ท่อนพันธุ์ด้วยสารป้องกันกำจัดแมลงหวี่ขาวยาสูบและเพลี้ยแป้ง 5–10 นาที",
                "ยกร่องสูง 30–40 ซม. เพื่อเพิ่มความโปร่งของดินให้หัวขยายตัวได้ดี",
                "ปักท่อนพันธุ์ในแนวดิ่งหรือเอียงเล็กน้อย ลึก 5–10 ซม."
            ],
            "key_cautions": "ตรวจแปลงสม่ำเสมอเพื่อเฝ้าระวังโรคใบด่างมันสำปะหลัง (Cassava Mosaic Disease) ที่ถ่ายทอดทางแมลงหวี่ขาว",
            "source_agency": "กรมวิชาการเกษตร"
        },
        "อ้อย": {
            "crop_group": "พืชไร่ / อ้อย",
            "planting_window": "อ้อยต้นฝน: เมษายน – พฤษภาคม | อ้อยปลายฝน: ตุลาคม – ธันวาคม",
            "harvest_window": "ธันวาคม – เมษายน (ตามฤดูเปิดหีบของโรงงานน้ำตาล)",
            "growth_duration_days": "10 – 14 เดือน (300 – 420 วัน)",
            "water_soil_req": "ดินร่วน ดินร่วนปนทราย หรือดินร่วนเหนียว ระบายน้ำดี ไม่ชอบน้ำขัง ต้องการน้ำ 1,200–1,500 มม./ปี ค่า pH 5.5–7.0",
            "pre_planting_checklist": [
                "ไถดะ ไถแปร และไถระเบิดดินดานลึก 40–50 ซม. เพื่อให้รากอ้อยหยั่งลึกทนแล้ง",
                "คัดเลือกท่อนพันธุ์อ้อยอายุ 8–10 เดือน มีตาสมบูรณ์ ปราศจากโรคขาวใบและหนอนกอ",
                "แช่ท่อนพันธุ์ในสารป้องกันกำจัดเชื้อราและแมลงก่อนปลูก",
                "ยกร่องลึก 25–30 ซม. ระยะระหว่างร่อง 1.3–1.5 เมตร เพื่อรองรับรถตัดอ้อย",
                "ใส่ปุ๋ยรองพื้นสูตร 16-16-8 หรือ 15-15-15 อัตรา 30–50 กก./ไร่"
            ],
            "key_cautions": "ระวังหนอนกออ้อยระบาดช่วงแล้ง และงดการเผาอ้อยก่อนเก็บเกี่ยวเพื่อรักษาคุณภาพดินและลดฝุ่น PM2.5",
            "source_agency": "สำนักงานคณะกรรมการอ้อยและน้ำตาลทราย และกรมวิชาการเกษตร"
        },
        "ยางพารา": {
            "crop_group": "ไม้ยืนต้น",
            "planting_window": "พฤษภาคม – กรกฎาคม (ช่วงฤดูฝนชุก ดินมีความชื้นต่อเนื่อง)",
            "harvest_window": "เริ่มเปิดกรีดเมื่ออายุ 7 ปี (เส้นรอบวงต้น 50 ซม.) กรีดได้ตลอดปียกเว้นช่วงผลัดใบ",
            "growth_duration_days": "ไม้ยืนต้นอายุ 25–30 ปี",
            "water_soil_req": "ดินลึกไม่น้อยกว่า 1 เมตร ระบายน้ำดี ไม่มีชั้นหินตื้น ปริมาณฝน 1,500–2,000 มม./ปี",
            "pre_planting_checklist": [
                "ขุดหลุมขนาด 50x50x50 ซม. รองก้นหลุมด้วยปุ๋ยร็อคฟอสเฟต",
                "เลือกต้นพันธุ์ชำถุงสมบูรณ์ ยอดแก่ใบแก่",
                "ปลูกพืชคลุมดินตระกูลถั่วเพื่อรักษาความชื้นและเพิ่มไนโตรเจน",
                "ทำแนวกันไฟรอบสวนในช่วงฤดูแล้ง"
            ],
            "key_cautions": "ระวังโรคใบร่วงชนิดใหม่ในยางพารา และหลีกเลี่ยงการเปิดกรีดในวันที่ฝนตกเพื่อป้องกันเชื้อรารากและเปลือกเน่า",
            "source_agency": "การยางแห่งประเทศไทย (กยท.)"
        },
        "ปาล์มน้ำมัน": {
            "crop_group": "ไม้ยืนต้น",
            "planting_window": "พฤษภาคม – สิงหาคม",
            "harvest_window": "เริ่มเก็บเกี่ยวได้เมื่ออายุ 2.5–3 ปี เก็บผลสุกทุก 15–20 วันตลอดปี",
            "growth_duration_days": "ไม้ยืนต้นอายุ 20–25 ปี",
            "water_soil_req": "ต้องการน้ำสม่ำเสมอ 1,800–2,500 มม./ปี ฝนตกสม่ำเสมอ แสงแดดจัดอย่างน้อย 5 ชม./วัน",
            "pre_planting_checklist": [
                "เลือกพันธุ์ปาล์มน้ำมันเทเนอรา (Tenera) ที่ได้รับการรับรองจากกรมวิชาการเกษตร",
                "วางระยะปลูกสามเหลี่ยมด้านเท่า 9x9x9 เมตร (22 ต้น/ไร่)",
                "รองก้นหลุมด้วยปุ๋ยฟอสเฟตและโดโลไมท์ปรับสภาพกรดของดิน",
                "จัดการระบบระบายน้ำในสวนปาล์มน้ำมันไม่ให้น้ำท่วมขังเกิน 7 วัน"
            ],
            "key_cautions": "ขาดน้ำนานเกิน 2 เดือนจะทำให้ทะลายฝ่อ ดอกตัวเมียลดลงอย่างมีนัยสำคัญ",
            "source_agency": "กรมวิชาการเกษตร"
        },
        "ทุเรียน": {
            "crop_group": "ไม้ผลยืนต้น",
            "planting_window": "ต้นฤดูฝน: พฤษภาคม – มิถุนายน",
            "harvest_window": "เมษายน – กรกฎาคม (ขึ้นกับพื้นที่ ภาคตะวันออก / ภาคใต้)",
            "growth_duration_days": "เริ่มให้ผลผลิตเมื่ออายุ 4–5 ปี",
            "water_soil_req": "ดินร่วนซุย ระบายน้ำดีเยี่ยม ไม่ทนน้ำท่วมขัง ค่า pH 5.5–6.5 ต้องมีระบบน้ำสปริงเกลอร์ประจำต้น",
            "pre_planting_checklist": [
                "ยกร่องหรือพูนโคกสูง 50–80 ซม. เพื่อป้องกันโรครากเน่าโคนเน่า",
                "ติดตั้งระบบน้ำมินิสปริงเกลอร์ให้พร้อมก่อนลงปลูก",
                "ทำร่มเงาพรางแสงแดดให้ต้นกล้าในช่วง 1–2 ปีแรก",
                "พ่นสารชีวภัณฑ์ไตรโคเดอร์มาป้องกันเชื้อรา Phytophthora ตั้งแต่เริ่มปลูก"
            ],
            "key_cautions": "ไวต่อโรคไฟทอปโทรา (รากเน่าโคนเน่า) และห้ามให้น้ำขังโคนต้นเด็ดขาด",
            "source_agency": "กรมส่งเสริมการเกษตร"
        }
    }

    # Default fallback for crops not explicitly specialized
    default_guide = {
        "crop_group": "พืชผลการเกษตร",
        "planting_window": "ต้นฤดูฝน (พฤษภาคม – มิถุนายน) หรือตามระบบน้ำชลประทาน",
        "harvest_window": "ตามรอบอายุการเก็บเกี่ยวของชนิดพืช",
        "growth_duration_days": "ขึ้นอยู่กับสายพันธุ์และฤดูกาล",
        "water_soil_req": "ดินร่วนระบายน้ำดี ค่า pH 5.5–6.5 และมีแหล่งน้ำสนับสนุนเพียงพอ",
        "pre_planting_checklist": [
            "ตรวจวิเคราะห์ความเป็นกรด-ด่างและธาตุอาหารในดินก่อนเพาะปลูก",
            "จัดเตรียมแหล่งน้ำสำรองและวางระบบให้น้ำที่เหมาะสม",
            "คัดเลือกสายพันธุ์ที่มีคุณภาพและผ่านการรับรองมาตรฐานจากทางราชการ",
            "เตรียมดิน ไถพรวนตากดิน 7–14 วันเพื่อลดการสะสมของโรคและแมลงศัตรูพืช",
            "ติดต่อสำนักงานเกษตรอำเภอหรือหมอดินอาสาในพื้นที่เพื่อขอคำแนะนำเฉพาะทาง"
        ],
        "key_cautions": "ติดตามการแจ้งเตือนสภาพอากาศและปริมาณน้ำฝนจากกรมอุตุนิยมวิทยาและ สสน. อย่างสม่ำเสมอ",
        "source_agency": "กรมส่งเสริมการเกษตร กระทรวงเกษตรและสหกรณ์"
    }

    if product in guides:
        return guides[product]
    for k, g in guides.items():
        if k in product or product in k:
            return g
    return default_guide



def join_climate_agriculture_nationwide(
    df_ag: pd.DataFrame,
    df_rain_annual: pd.DataFrame,
    df_temp_annual: Optional[pd.DataFrame] = None,
    year: Optional[int] = None,
    product: Optional[str] = None
) -> pd.DataFrame:
    """
    Join HII provincial annual rainfall (and temperature) with provincial agricultural records.
    Filters exclusively for provincial level records to prevent double counting.
    """
    ag_prov = df_ag[df_ag["geo_level"] == "province"].copy()
    if year:
        ag_prov = ag_prov[ag_prov["year"] == year]
    if product and product != "ทั้งหมด":
        ag_prov = ag_prov[ag_prov["product"] == product]
        
    joined = pd.merge(
        ag_prov,
        df_rain_annual[["province", "year", "rainfall_annual_mm", "data_source_type"]],
        on=["province", "year"],
        how="inner"
    )
    
    if df_temp_annual is not None and not df_temp_annual.empty:
        joined = pd.merge(
            joined,
            df_temp_annual[["province", "year", "temp_annual_max", "temp_annual_min", "temp_annual_mean"]],
            on=["province", "year"],
            how="left"
        )
        
    return joined


def calculate_bivariate_stats(
    x: pd.Series,
    y: pd.Series
) -> Dict[str, Any]:
    """
    Calculate Pearson and Spearman correlation coefficients and p-values.
    Safely handles missing values, low sample size, and zero-variance.
    """
    valid_df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(valid_df)
    
    if n < 3:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "is_valid": False,
            "message": f"จำนวนคู่ข้อมูลมีเพียง {n} จุด ซึ่งไม่เพียงพอสำหรับการวิเคราะห์สหสัมพันธ์ (ต้องการอย่างน้อย 3 จุด)"
        }
        
    x_clean = valid_df["x"]
    y_clean = valid_df["y"]
    
    if x_clean.std() == 0 or y_clean.std() == 0:
        return {
            "n": n,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "is_valid": False,
            "message": "ตัวแปรตัวใดตัวหนึ่งมีค่าคงที่ (ความแปรปรวนเป็นศูนย์) ไม่สามารถคำนวณสหสัมพันธ์ได้"
        }
        
    pr, pp = stats.pearsonr(x_clean, y_clean)
    sr, sp = stats.spearmanr(x_clean, y_clean)
    
    return {
        "n": n,
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "is_valid": True,
        "message": "คำนวณสำเร็จ"
    }


def generate_climate_interpretation(
    bivariate_result: Dict[str, Any],
    crop_name: str,
    climate_var_name: str,
    ag_var_name: str,
    geo_scope: str = "77 จังหวัดทั่วประเทศไทย",
    year_range: str = "2561–2562"
) -> str:
    """
    Generate an academic, data-driven interpretation of the climate-agriculture relationship.
    Avoids unfounded causal assertions and explicitly mentions statistical caveats.
    """
    if not bivariate_result.get("is_valid", False):
        return f"**ข้อจำกัดของข้อมูล:** {bivariate_result.get('message', 'ข้อมูลไม่เพียงพอสำหรับการสรุปผล')}"
        
    r = bivariate_result["pearson_r"]
    p = bivariate_result["pearson_p"]
    n = bivariate_result["n"]
    
    # Strength categorization
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "ระดับสูง (Strong)"
    elif abs_r >= 0.4:
        strength = "ระดับปานกลาง (Moderate)"
    elif abs_r >= 0.2:
        strength = "ระดับต่ำ (Weak)"
    else:
        strength = "น้อยมากหรือแทบไม่มีความสัมพันธ์ (Negligible)"
        
    direction = "เชิงบวก (Positive)" if r > 0 else "เชิงลบ (Negative)"
    significance = "มีนัยสำคัญทางสถิติที่ระดับ 0.05 (p < 0.05)" if p < 0.05 else "ยังไม่มีนัยสำคัญทางสถิติที่ระดับ 0.05 (p ≥ 0.05)"
    
    text = (
        f"**ผลการวิเคราะห์ทางสถิติ:**\n"
        f"- จากการวิเคราะห์ข้อมูลจำนวน **{n} ปีคู่ข้อมูล** ในช่วงปี **พ.ศ. {year_range}** ({geo_scope}) สำหรับ **{crop_name}**\n"
        f"- ค่าสัมประสิทธิ์สหสัมพันธ์แบบเพียร์สัน ($r$) ระหว่าง **{climate_var_name}** กับ **{ag_var_name}** เท่ากับ **{r:.3f}** (p-value = **{p:.4f}**)\n"
        f"- บ่งชี้ความสัมพันธ์{direction}ใน{strength} โดยความสัมพันธ์นี้{significance}\n\n"
        f"**ข้อควรระวังทางวิชาการและข้อจำกัด:**\n"
        f"1. **Correlation ไม่ใช่ Causation:** ความสัมพันธ์ทางสถิตินี้ไม่ได้พิสูจน์ว่า {climate_var_name} เป็นสาเหตุโดยตรงของการเปลี่ยนแปลง {ag_var_name} เนื่องจากผลผลิตทางการเกษตรยังขึ้นอยู่กับระบบชลประทาน เทคโนโลยีการเพาะปลูก ปุ๋ย โรคแมลง และปัจจัยทางเศรษฐกิจ\n"
        f"2. **ขนาดกลุ่มตัวอย่าง:** ข้อมูลมีจำนวน {n} จุด การสรุปผลควรใช้เป็นสมมติฐานเบื้องต้นเพื่อการศึกษาต่อยอดร่วมกับผู้เชี่ยวชาญด้านปฐพีวิทยาและพืชศาสตร์ต่อไป"
    )
    return text


def classify_area_yield_quadrants(
    df: pd.DataFrame,
    area_col: str = "area_plant_rai",
    yield_col: str = "calc_yield_per_rai",
    label_col: str = "product"
) -> Tuple[pd.DataFrame, float, float]:
    """
    Classify items into 4 quadrants based on median Planted Area and median Yield:
    - Q1 (High Area, High Yield): พื้นที่ปลูกมาก & ผลผลิตต่อไร่สูง (เสาหลักการผลิต)
    - Q2 (Low Area, High Yield): พื้นที่ปลูกน้อย & ผลผลิตต่อไร่สูง (พืชผลตอบแทนต่อไร่สูง / พืชเฉพาะทาง)
    - Q3 (Low Area, Low Yield): พื้นที่ปลูกน้อย & ผลผลิตต่อไร่ต่ำ (พืชขนาดเล็ก / ความเสี่ยงผลตอบแทนต่ำ)
    - Q4 (High Area, Low Yield): พื้นที่ปลูกมาก & ผลผลิตต่อไร่ต่ำ (พื้นที่กว้างแต่ Yield ต่ำ / มีโอกาสพัฒนาเพิ่มประสิทธิภาพ)
    """
    clean_df = df.dropna(subset=[area_col, yield_col]).copy()
    if clean_df.empty:
        return clean_df, 0.0, 0.0
        
    area_median = float(clean_df[area_col].median())
    yield_median = float(clean_df[yield_col].median())
    
    def assign_quad(row):
        is_high_area = row[area_col] >= area_median
        is_high_yield = row[yield_col] >= yield_median
        if is_high_area and is_high_yield:
            return "Q1: พื้นที่มาก - Yield สูง"
        elif not is_high_area and is_high_yield:
            return "Q2: พื้นที่น้อย - Yield สูง"
        elif not is_high_area and not is_high_yield:
            return "Q3: พื้นที่น้อย - Yield ต่ำ"
        else:
            return "Q4: พื้นที่มาก - Yield ต่ำ"
            
    clean_df["quadrant"] = clean_df.apply(assign_quad, axis=1)
    return clean_df, area_median, yield_median


def get_top_provinces(
    df: pd.DataFrame,
    metric: str = "production_ton",
    top_n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get top N provinces by specified agricultural metric, strictly excluding aggregates.
    """
    prov_df = df[df["geo_level"] == "province"].copy()
    if prov_df.empty or metric not in prov_df.columns:
        return pd.DataFrame()
        
    # If grouped by province
    grouped = prov_df.groupby("province").agg(
        production_ton=("production_ton", "sum"),
        area_plant_rai=("area_plant_rai", "sum"),
        area_harvest_rai=("area_harvest_rai", "sum")
    ).reset_index()
    
    # Calculate weighted yield and harvest ratio
    grouped["calc_yield_per_rai"] = np.where(
        grouped["area_harvest_rai"] > 0,
        grouped["production_ton"] * 1000.0 / grouped["area_harvest_rai"],
        np.nan
    )
    grouped["calc_harvest_ratio"] = np.where(
        grouped["area_plant_rai"] > 0,
        grouped["area_harvest_rai"] / grouped["area_plant_rai"] * 100.0,
        np.nan
    )
    
    res = grouped.sort_values(by=metric, ascending=ascending).head(top_n).reset_index(drop=True)
    return res


def get_top_crops(
    df: pd.DataFrame,
    metric: str = "production_ton",
    top_n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get top N crops by specified agricultural metric, strictly avoiding double counting.
    """
    # Use provincial records sum to ensure consistency
    prov_df = df[df["geo_level"] == "province"].copy()
    if prov_df.empty or metric not in prov_df.columns:
        return pd.DataFrame()
        
    grouped = prov_df.groupby("product").agg(
        production_ton=("production_ton", "sum"),
        area_plant_rai=("area_plant_rai", "sum"),
        area_harvest_rai=("area_harvest_rai", "sum")
    ).reset_index()
    
    grouped["calc_yield_per_rai"] = np.where(
        grouped["area_harvest_rai"] > 0,
        grouped["production_ton"] * 1000.0 / grouped["area_harvest_rai"],
        np.nan
    )
    grouped["calc_harvest_ratio"] = np.where(
        grouped["area_plant_rai"] > 0,
        grouped["area_harvest_rai"] / grouped["area_plant_rai"] * 100.0,
        np.nan
    )
    
    res = grouped.sort_values(by=metric, ascending=ascending).head(top_n).reset_index(drop=True)
    return res
