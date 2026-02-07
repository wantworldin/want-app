import streamlit as st
from PIL import Image
import numpy as np
import cv2
import math
import random
from datetime import datetime

# ==============================================================================
# [WA Platform Ver 6.0] Global Standard Edition
# 1. Multi-language (KR, EN, CN)
# 2. Safe Number Simulation (Privacy Protection)
# 3. WA Branding
# ==============================================================================

# --- [1] 다국어 사전 (Language Dictionary) ---
LANG = {
    "KR": {
        "title": "WA 플랫폼 (Want Appraiser)",
        "sidebar_title": "언어 설정 (Language)",
        "tab1": "🎨 미술품(Art)",
        "tab2": "🚗 주차(Car)",
        "tab3": "🧸 사물/미아(Object)",
        "reg_title": "등록 (Register)",
        "ver_title": "검증 (Verify)",
        "upload_org": "원본 이미지 업로드",
        "upload_ver": "검증할 이미지 업로드",
        "name_input": "작품명/소유자",
        "car_input": "차량 번호",
        "obj_input": "이름/연락처 (실제 번호)",
        "btn_reg": "등록하기",
        "btn_ver": "검증하기",
        "btn_call": "📞 안심번호로 전화걸기",
        "mode_strict": "🕵️ S급 모사품 감별 (초정밀)",
        "success_gen": "🎉 진품입니다!",
        "fail_gen": "🚨 가품/불일치",
        "info_score": "점수",
        "info_ratio": "일치율",
        "safe_num_msg": "안심번호가 생성되었습니다:",
        "calling_msg": "안심번호로 연결 중입니다...",
        "reg_success": "등록이 완료되었습니다.",
        "err_no_data": "등록된 데이터가 없습니다.",
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
        "ver_title": "Verify",
        "upload_org": "Upload Original Image",
        "upload_ver": "Upload Image to Verify",
        "name_input": "Artwork Name / Owner",
        "car_input": "License Plate",
        "obj_input": "Name / Phone (Real)",
        "btn_reg": "Register",
        "btn_ver": "Verify",
        "btn_call": "📞 Call via Safe Number",
        "mode_strict": "🕵️ Forensic Mode (Strict)",
        "success_gen": "🎉 Authentic / Match Found!",
        "fail_gen": "🚨 Fake / No Match",
        "info_score": "Score",
        "info_ratio": "Ratio",
        "safe_num_msg": "Safe Number Generated:",
        "calling_msg": "Calling via Safe Number...",
        "reg_success": "Registration Complete.",
        "err_no_data": "No data registered.",
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
        "ver_title": "验证",
        "upload_org": "上传原始图片",
        "upload_ver": "上传验证图片",
        "name_input": "作品名称 / 所有者",
        "car_input": "车牌号码",
        "obj_input": "姓名 / 电话 (真实)",
        "btn_reg": "注册",
        "btn_ver": "验证",
        "btn_call": "📞 拨打虚拟号码",
        "mode_strict": "🕵️ 超精密鉴别模式",
        "success_gen": "🎉 正品 / 匹配成功!",
        "fail_gen": "🚨 赝品 / 不匹配",
        "info_score": "分数",
        "info_ratio": "匹配率",
        "safe_num_msg": "已生成虚拟号码:",
        "calling_msg": "正在通过虚拟号码连接...",
        "reg_success": "注册完成。",
        "err_no_data": "没有注册数据。",
        "calc_fee": "停车费",
        "parking_time": "停车时间",
        "min": "分"
    }
}

# --- [2] 엔진 (Ver 4.3 Core) ---
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

# 언어 선택 사이드바
with st.sidebar:
    st.title("🌐 Language")
    lang_code = st.radio("Select Language", ["KR", "EN", "CN"])
    
txt = LANG[lang_code] # 선택된 언어 팩 로드

st.title(f"📱 {txt['title']}")

# 안심번호 생성기 (데모용)
def get_safe_number():
    return f"0505-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

tab1, tab2, tab3 = st.tabs([txt['tab1'], txt['tab2'], txt['tab3']])

# 1. 미술품 탭
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
        if ver and st.button(txt['btn_ver']):
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
                st.success(f"{txt['success_gen']} ({item['name']})")
                st.write(f"{txt['info_score']}: {c} / {txt['info_ratio']}: {r:.1f}%")
                st.image(img, use_container_width=True)
            else: st.error(txt['fail_gen'])

# 2. 주차 탭 (안심번호 추가)
with tab2:
    c3, c4 = st.columns(2)
    with c3:
        st.subheader(txt['reg_title'])
        with st.form("car_reg", clear_on_submit=True):
            up = st.file_uploader(txt['upload_org'], key="c_up")
            no = st.text_input(txt['car_input'])
            # 실제 번호 입력받지만, 내부적으로 안심번호 생성
            phone = st.text_input(txt['obj_input']) 
            if st.form_submit_button(txt['btn_reg']) and up:
                safe_num = get_safe_number()
                st.session_state['cars'].append({
                    "image": Image.open(up), "no": no, "phone": safe_num, "time": datetime.now()
                })
                st.success(f"{txt['reg_success']} ({txt['safe_num_msg']} {safe_num})")
    with c4:
        st.subheader(txt['ver_title'])
        ver = st.file_uploader(txt['upload_ver'], key="c_ver")
        if ver and st.button(txt['btn_ver']):
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
                st.success(f"{txt['success_gen']} : {item['no']}")
                st.info(f"{txt['parking_time']}: {duration.seconds//60}{txt['min']} / {txt['calc_fee']}: {fee:,}")
                
                # 안심번호 통화 버튼
                st.markdown("---")
                st.write(f"📞 **{txt['safe_num_msg']} {item['phone']}**")
                if st.button(txt['btn_call'], key="call_car"):
                    st.toast(f"{txt['calling_msg']} ({item['phone']})")
            else: st.error(txt['fail_gen'])

# 3. 사물 탭 (안심번호 추가)
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
        if ver and st.button(txt['btn_ver']):
            t_img = Image.open(ver)
            if not st.session_state['objects']: st.error(txt['err_no_data']); st.stop()
            best = (None, 0, 0)
            for item in st.session_state['objects']:
                is_g, c, r, _ = match_engine(item['image'], t_img, "fast")
                if c > best[1]: best = (item, c, r)
            item, c, r = best
            if item and c >= 10 and r >= 15.0:
                st.success(f"{txt['success_gen']}")
                st.info(f"Owner: {item['info']}")
                
                # 안심번호 통화 버튼
                st.markdown("---")
                st.write(f"📞 **{txt['safe_num_msg']} {item['phone']}**")
                if st.button(txt['btn_call'], key="call_obj"):
                    st.toast(f"{txt['calling_msg']} ({item['phone']})")
            else: st.error(txt['fail_gen'])
