import streamlit as st
from PIL import Image
import numpy as np
import cv2
import math
import random
from datetime import datetime

# ==============================================================================
# [WA Platform Ver 6.2] Final Optimized Edition
# 1. 주차 탭: "순수 정산 시스템"으로 복구 (전화 기능 삭제, Ver 5.0 스타일)
# 2. 사물/미아 탭: "주인 찾기 & 안심번호" 특화 (요청하신 UI 적용)
# 3. 미술품 탭: S급 감별 유지
# ==============================================================================

# --- [1] 다국어 사전 (Language Dictionary) ---
LANG = {
    "KR": {
        "title": "WA 플랫폼 (Want Appraiser)",
        "sidebar_title": "언어 설정 (Language)",
        "tab1": "🎨 미술품(Art)",
        "tab2": "🚗 주차(Car)",
        "tab3": "🧸 사물/미아(Object)",
        
        # 공통
        "reg_title": "등록 (Register)",
        "ver_title": "검증 (Verify)",
        "upload_org": "원본 이미지 업로드",
        "upload_ver": "이미지 업로드",
        "btn_reg": "등록하기",
        "reg_success": "등록이 완료되었습니다.",
        "err_no_data": "등록된 데이터가 없습니다.",
        
        # 탭별 입력창
        "name_input": "작품명/소유자",
        "car_input": "차량 번호",
        "obj_input": "이름/연락처 (실제 번호)",
        
        # 버튼
        "btn_ver_art": "검증하기",
        "btn_ver_car": "출차/정산 요청", # 주차 전용
        "btn_find_owner": "주인찾기",    # 사물 전용
        
        # 메시지
        "mode_strict": "🕵️ S급 모사품 감별 (초정밀)",
        "success_art": "🎉 진품입니다!",
        "success_car": "✅ 차량 인식 성공",
        "fail_gen": "🚨 데이터 불일치 / 정보 없음",
        
        "info_score": "점수",
        "info_ratio": "일치율",
        
        # 안심번호 (사물 탭 전용)
        "safe_num_msg": "안심번호 생성:",
        "owner_contact": "소유자 연락처: 안심번호",
        "btn_call_simple": "📞 전화걸기",
        "calling_msg": "연결 중입니다...",
        
        # 주차 요금
        "calc_fee": "주차 요금",
        "parking_time": "주차 시간",
        "min": "분"
    },
    "EN": {
        "title": "WA Platform (Want Appraiser)",
        "sidebar_title": "Language Settings",
        "tab1": "🎨 Art",
        "tab2": "🚗 Car",
        "tab3": "🧸 Object",
        "reg_title": "Register",
        "ver_title": "Verify / Find",
        "upload_org": "Upload Original",
        "upload_ver": "Upload Image",
        "btn_reg": "Register",
        "reg_success": "Registration Complete.",
        "err_no_data": "No data found.",
        "name_input": "Artwork Name / Owner",
        "car_input": "License Plate",
        "obj_input": "Name / Phone (Real)",
        "btn_ver_art": "Verify",
        "btn_ver_car": "Check Out",
        "btn_find_owner": "Find Owner",
        "mode_strict": "🕵️ Forensic Mode",
        "success_art": "🎉 Authentic!",
        "success_car": "✅ Car Identified",
        "fail_gen": "🚨 No Match Found",
        "info_score": "Score",
        "info_ratio": "Ratio",
        "safe_num_msg": "Safe #:",
        "owner_contact": "Owner Contact (Safe #):",
        "btn_call_simple": "📞 Call",
        "calling_msg": "Calling...",
        "calc_fee": "Fee",
        "parking_time": "Duration",
        "min": "min"
    },
    "CN": {
        "title": "WA 平台 (Want Appraiser)",
        "sidebar_title": "语言设置",
        "tab1": "🎨 艺术品",
        "tab2": "🚗 停车",
        "tab3": "🧸 寻物/寻人",
        "reg_title": "注册",
        "ver_title": "验证 / 寻找",
        "upload_org": "上传原始图片",
        "upload_ver": "上传图片",
        "btn_reg": "注册",
        "reg_success": "注册完成。",
        "err_no_data": "没有数据。",
        "name_input": "作品名称 / 所有者",
        "car_input": "车牌号码",
        "obj_input": "姓名 / 电话 (真实)",
        "btn_ver_art": "验证",
        "btn_ver_car": "结算请求",
        "btn_find_owner": "寻找失主",
        "mode_strict": "🕵️ 精密鉴别模式",
        "success_art": "🎉 正品!",
        "success_car": "✅ 车辆识别成功",
        "fail_gen": "🚨 不匹配",
        "info_score": "分数",
        "info_ratio": "匹配率",
        "safe_num_msg": "虚拟号码:",
        "owner_contact": "失主联系方式 (虚拟号):",
        "btn_call_simple": "📞 拨打电话",
        "calling_msg": "正在连接...",
        "calc_fee": "停车费",
        "parking_time": "停车时间",
        "min": "分"
    }
}

# --- [2] 엔진 (Ver 4.3 Core 유지) ---
def resize_optimized(img_array, max_dim):
    h, w = img_array.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_array

def calculate_angles(pt1, pt2, pt3):
    def length(p1, p2): return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    a, b, c = length(pt2, pt3), length(pt1, pt3), length(pt1, pt2)
    if a==0 or b==0 or c==0: return [0,0,0]
    try:
        angle_A = np.degrees(np.arccos(np.clip((b**2+c**2-a**2)/(2*b*c), -1.0, 1.0)))
        angle_B = np.degrees(np.arccos(np.clip((a**2+c**2-b**2)/(2*a*c), -1.0, 1.0)))
        angle_C = 180 - angle_A - angle_B
    except: return [0,0,0]
    return sorted([angle_A, angle_B, angle_C])

def verify_geometry(kp1, kp2, good_matches, strict_mode):
    ransac_thresh = 1.0 if strict_mode else 4.0
    angle_thresh = 1.0 if strict_mode else 3.0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    if len(good_matches) < 4: return []
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if M is None: return []
    matches_mask = mask.ravel().tolist()
    global_correct = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
    final_indices = set()
    check_list = global_correct[:200]
    for i in range(len(check_list) - 2):
        m1, m2, m3 = check_list[i], check_list[i+1], check_list[i+2]
        p1, p2, p3 = kp1[m1.queryIdx].pt, kp1[m2.queryIdx].pt, kp1[m3.queryIdx].pt
        q1, q2, q3 = kp2[m1.trainIdx].pt, kp2[m2.trainIdx].pt, kp2[m3.trainIdx].pt
        ang1, ang2 = calculate_angles(p1, p2, p3), calculate_angles(q1, q2, q3)
        if sum([abs(a-b) for a,b in zip(ang1, ang2)]) < angle_thresh:
            final_indices.update([m1, m2, m3])
    return list(final_indices)

def match_engine(img1_pil, img2_pil, mode="forensic", strict=False):
    max_dim = 2000 if mode == "forensic" else 640
    n_features = 10000 if mode == "forensic" else 1000
    scales = [0.5, 1.0] if mode == "forensic" else [1.0]
    img1 = resize_optimized(np.array(img1_pil), max_dim)
    img2 = resize_optimized(np.array(img2_pil), max_dim)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des2) < 5: return False, 0, 0, None, "Err"
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))
    best_res = (0, 0.0, None)
    for scale in scales:
        try:
            if scale == 1.0: r_gray2, r_img2 = gray2, img2
            else:
                h, w = gray2.shape
                r_gray2 = cv2.resize(gray2, (int(w*scale), int(h*scale)))
                r_img2 = cv2.resize(img2, (int(w*scale), int(h*scale)))
                _, des2 = sift.detectAndCompute(r_gray2, None)
            if des2 is None or len(des2) < 5: continue
            matches = flann.knnMatch(des1, des2, k=2)
            ratio_thresh = 0.7 if strict else 0.75
            good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
            if mode == "forensic":
                final = verify_geometry(kp1, sift.detect(r_gray2, None), good, strict)
            else:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([sift.detect(r_gray2, None)[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                if len(good) < 4: continue
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                final = [good[i] for i in range(len(good)) if mask.ravel()[i]]
            cnt = len(final)
            ratio = (cnt / len(des2) * 100) if len(des2) > 0 else 0
            if cnt > best_res[0]:
                res_img = cv2.drawMatches(img1, kp1, r_img2, sift.detect(r_gray2,None), final, None, flags=2)
                best_res = (cnt, ratio, cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB))
                if mode == "forensic" and ratio > 15 and cnt > 200: break
                if mode == "fast" and ratio > 15: break
        except: continue
    count, ratio, img = best_res
    is_genuine = False
    if mode == "forensic":
        if strict: is_genuine = (count >= 50 and ratio >= 3.0)
        else: is_genuine = (count >= 80) or (count >= 15 and ratio >= 10.0)
        if ratio < 1.0: is_genuine = False
    else: is_genuine = (count >= 10 and ratio >= 15.0)
    return is_genuine, count, ratio, img

# --- [3] UI 및 로직 ---
st.set_page_config(page_title="WA Platform", layout="wide", page_icon="🌐")

if 'artworks' not in st.session_state: st.session_state['artworks'] = [] 
if 'cars' not in st.session_state: st.session_state['cars'] = []
if 'objects' not in st.session_state: st.session_state['objects'] = []

with st.sidebar:
    st.title("🌐 Language")
    lang_code = st.radio("Select Language", ["KR", "EN", "CN"])
    
txt = LANG[lang_code]

st.title(f"📱 {txt['title']}")

def get_safe_number():
    return f"0505-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

tab1, tab2, tab3 = st.tabs([txt['tab1'], txt['tab2'], txt['tab3']])

# 1. 미술품 탭 (Forensic Mode)
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(txt['reg_title'])
        with st.form("art_reg", clear_on_submit=True):
            up = st.file_uploader(txt['upload_org'], key="a_up")
            name = st.text_input(txt['name_input'])
            if st.form_submit_button(txt['btn_reg']) and up:
                st.session_state['artworks'].append({"image": Image.open(up), "name": name})
                st.success(txt['reg_success'])
    with c2:
        st.subheader(txt['ver_title'])
        ver = st.file_uploader(txt['upload_ver'], key="a_ver")
        strict = st.checkbox(txt['mode_strict'], key="strict")
        if ver and st.button(txt['btn_ver_art']):
            t_img = Image.open(ver)
            if not st.session_state['artworks']: st.error(txt['err_no_data']); st.stop()
            best = (None, 0, 0, None)
            for item in st.session_state['artworks']:
                is_g, c, r, img = match_engine(item['image'], t_img, "forensic", strict)
                if c > best[1]: best = (item, c, r, img)
            item, c, r, img = best
            is_genuine = False
            if strict: is_genuine = (c >= 50 and r >= 3.0)
            else: is_genuine = (c >= 80) or (c >= 15 and r >= 10.0)
            if r < 1.0: is_genuine = False

            if item and is_genuine:
                st.success(f"{txt['success_art']} ({item['name']})")
                st.write(f"{txt['info_score']}: {c} / {txt['info_ratio']}: {r:.1f}%")
                st.image(img, use_container_width=True)
            else: st.error(txt['fail_gen'])

# 2. 주차 탭 (복구: 전화 기능 삭제, 순수 정산 모드)
with tab2:
    c3, c4 = st.columns(2)
    with c3:
        st.subheader(txt['reg_title'])
        with st.form("car_reg", clear_on_submit=True):
            up = st.file_uploader(txt['upload_org'], key="c_up")
            no = st.text_input(txt['car_input'])
            # [복구] 연락처 입력 삭제 (안심번호 미사용)
            if st.form_submit_button(txt['btn_reg']) and up:
                st.session_state['cars'].append({
                    "image": Image.open(up), "no": no, "time": datetime.now()
                })
                st.success(txt['reg_success'])
    with c4:
        st.subheader(txt['ver_title'])
        ver = st.file_uploader(txt['upload_ver'], key="c_ver")
        if ver and st.button(txt['btn_ver_car']):
            t_img = Image.open(ver)
            if not st.session_state['cars']: st.error(txt['err_no_data']); st.stop()
            best = (None, 0, 0)
            for item in st.session_state['cars']:
                is_g, c, r, _ = match_engine(item['image'], t_img, "fast")
                if c > best[1]: best = (item, c, r)
            item, c, r = best
            if item and c >= 10 and r >= 15.0:
                duration = datetime.now() - item['time']
                fee = (duration.seconds // 60 // 10) * 1000
                st.success(f"{txt['success_car']} : {item['no']}")
                st.info(f"{txt['parking_time']}: {duration.seconds//60}{txt['min']} / {txt['calc_fee']}: {fee:,}")
                # [복구] 전화 걸기 버튼 삭제
            else: st.error(txt['fail_gen'])

# 3. 사물/미아 탭 (요청사항 적용: 주인찾기 & 안심번호 특화)
with tab3:
    c5, c6 = st.columns(2)
    with c5:
        st.subheader(txt['reg_title'])
        with st.form("obj_reg", clear_on_submit=True):
            up = st.file_uploader(txt['upload_org'], key="o_up")
            info = st.text_input(txt['obj_input'])
            if st.form_submit_button(txt['btn_reg']) and up:
                safe_num = get_safe_number()
                st.session_state['objects'].append({"image": Image.open(up), "info": info, "phone": safe_num})
                st.success(f"{txt['reg_success']} ({txt['safe_num_msg']} {safe_num})")
    with c6:
        st.subheader(txt['ver_title'])
        ver = st.file_uploader(txt['upload_ver'], key="o_ver")
        
        # 버튼: 주인찾기
        if ver and st.button(txt['btn_find_owner']):
            t_img = Image.open(ver)
            if not st.session_state['objects']: st.error(txt['err_no_data']); st.stop()
            best = (None, 0, 0)
            for item in st.session_state['objects']:
                is_g, c, r, _ = match_engine(item['image'], t_img, "fast")
                if c > best[1]: best = (item, c, r)
            item, c, r = best
            
            if item and c >= 10 and r >= 15.0:
                # '진품입니다' 삭제 -> 바로 이름 표시
                st.success(f"✅ {item['info']}") 
                
                st.markdown("---")
                # 문구: 소유자 연락처: 안심번호 0505...
                st.subheader(f"{txt['owner_contact']} {item['phone']}")
                
                # 버튼: 전화걸기
                if st.button(txt['btn_call_simple'], key="call_obj"):
                    st.toast(f"{txt['calling_msg']} ({item['phone']})")
            else: st.error(txt['fail_gen'])

