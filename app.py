import streamlit as st
from PIL import Image
import numpy as np
import cv2
import math
from datetime import datetime

# ==============================================================================
# [WES Final Ver 4.3] Forensic Mode (Anti-Forgery)
# 핵심 기능: 'S급 모사품'을 잡기 위한 초정밀 검증(Forensic) 옵션 추가
# 일반 모드는 유연하게, 감별 모드는 오차 1.0 미만으로 칼같이 차단
# ==============================================================================

def resize_optimized(img_array, max_dim):
    h, w = img_array.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_array

def calculate_angles(pt1, pt2, pt3):
    def length(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    a, b, c = length(pt2, pt3), length(pt1, pt3), length(pt1, pt2)
    if a == 0 or b == 0 or c == 0: return [0, 0, 0]
    try:
        val_A = (b**2 + c**2 - a**2) / (2 * b * c)
        val_B = (a**2 + c**2 - b**2) / (2 * a * c)
        val_A = max(-1.0, min(1.0, val_A))
        val_B = max(-1.0, min(1.0, val_B))
        angle_A = math.degrees(math.acos(val_A))
        angle_B = math.degrees(math.acos(val_B))
        angle_C = 180 - angle_A - angle_B
    except ValueError: return [0, 0, 0]
    return sorted([angle_A, angle_B, angle_C])

def verify_geometry(kp1, kp2, good_matches, strict_mode=False):
    """
    strict_mode=True일 경우: 모사품 감별을 위해 허용 오차를 극단적으로 줄임
    """
    pts1 = [kp1[m.queryIdx].pt for m in good_matches]
    pts2 = [kp2[m.trainIdx].pt for m in good_matches]
    final_indices = set()
    
    # 1. RANSAC 임계값 조정
    # 일반: 4.0 (유연함) / 감별: 1.0 (픽셀 단위 일치 요구)
    ransac_thresh = 1.0 if strict_mode else 4.0
    
    if len(good_matches) >= 4:
        src_pts = np.float32(pts1).reshape(-1, 1, 2)
        dst_pts = np.float32(pts2).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if M is None: return []
        matches_mask = mask.ravel().tolist()
        global_correct_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
    else: return []

    # 2. 각도 검증 임계값 조정
    # 일반: 3.0도 허용 / 감별: 1.0도 허용 (사람 손으로는 절대 못 맞춤)
    angle_thresh = 1.0 if strict_mode else 3.0
    
    check_list = global_correct_matches[:300]
    for i in range(len(check_list) - 2):
        m1, m2, m3 = check_list[i], check_list[i+1], check_list[i+2]
        p1, p2, p3 = kp1[m1.queryIdx].pt, kp1[m2.queryIdx].pt, kp1[m3.queryIdx].pt
        q1, q2, q3 = kp2[m1.trainIdx].pt, kp2[m2.trainIdx].pt, kp2[m3.trainIdx].pt
        ang1 = calculate_angles(p1, p2, p3)
        ang2 = calculate_angles(q1, q2, q3)
        diff = sum([abs(a - b) for a, b in zip(ang1, ang2)])
        
        if diff < angle_thresh:
            final_indices.add(m1); final_indices.add(m2); final_indices.add(m3)
            
    return list(final_indices)

def match_art_forensic(img1_pil, img2_pil, strict_mode):
    # 해상도: 감별 모드에서는 2K (2000px), 일반은 1280px도 충분하지만 안전하게 2000 통일
    img1_cv = resize_optimized(np.array(img1_pil), max_dim=2000)
    img2_cv = resize_optimized(np.array(img2_pil), max_dim=2000)
    
    gray1 = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray1 = clahe.apply(gray1); gray2 = clahe.apply(gray2)

    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.03, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    if des1 is None or len(des1) < 10: return False, 0, 0, None, "특징점 부족"

    scales = [0.5, 1.0] 
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=40))
    
    best_count = 0; best_ratio = 0.0; best_img = None; best_scale = 1.0

    for scale in scales:
        try:
            if scale == 1.0: resized_gray2 = gray2; resized_kp2_img = img2_cv
            else:
                new_w = int(gray2.shape[1] * scale); new_h = int(gray2.shape[0] * scale)
                if new_w < 50 or new_h < 50: continue
                resized_gray2 = cv2.resize(gray2, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized_kp2_img = cv2.resize(img2_cv, (new_w, new_h))

            kp2, des2 = sift.detectAndCompute(resized_gray2, None)
            total_target_kps = len(kp2)
            if des2 is None or total_target_kps < 10: continue

            matches = flann.knnMatch(des1, des2, k=2)
            # 감별 모드일 때는 Ratio test도 0.7로 강화 (아주 똑같은 것만 허용)
            ratio_thresh = 0.7 if strict_mode else 0.75
            good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
            
            # [핵심] strict_mode 전달
            final_matches = verify_geometry(kp1, kp2, good_matches, strict_mode)
            current_count = len(final_matches)
            current_ratio = (current_count / total_target_kps) * 100 if total_target_kps > 0 else 0

            if current_count > best_count:
                best_count = current_count; best_ratio = current_ratio; best_scale = scale
                res_img = cv2.drawMatches(img1_cv, kp1, resized_kp2_img, kp2, final_matches, None, flags=2, matchColor=(0, 255, 0))
                best_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                if best_ratio > 15.0 and best_count > 200: break
        except: continue

    is_genuine = False
    
    if strict_mode:
        # [감별 모드] 점수가 팍 깎이므로 기준을 조금 낮게 잡되, 통과했다는 것 자체가 대단한 것임
        if best_count >= 50 and best_ratio >= 3.0: is_genuine = True
    else:
        # [일반 모드] 기존 Ver 4.2 로직 (유연함)
        if best_count >= 80: is_genuine = True
        elif best_count >= 15 and best_ratio >= 10.0: is_genuine = True
    
    if best_ratio < 1.0: is_genuine = False # 안전장치

    mode_str = "S급 모사품 감별" if strict_mode else "일반 검증"
    msg = f"🛡️ [{mode_str}] {best_count}점 (매칭률 {best_ratio:.1f}%)"
    return is_genuine, best_count, best_ratio, best_img, msg

# --- 고속 엔진 (유지) ---
def match_fast_rapid(img1_pil, img2_pil):
    img1_cv = np.array(img1_pil); img2_cv = np.array(img2_pil)
    img1_small = resize_optimized(img1_cv, max_dim=640)
    img2_small = resize_optimized(img2_cv, max_dim=640)
    gray1 = cv2.cvtColor(img1_small, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_small, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=1000) 
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des2) < 5: return False, 0, 0, None, "특징점 부족"
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    final_matches = []
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is not None:
            matches_mask = mask.ravel().tolist()
            final_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i]]
    count = len(final_matches)
    ratio = (count / len(kp2)) * 100 if len(kp2) > 0 else 0
    res_img = cv2.drawMatches(img1_small, kp1, img2_small, kp2, final_matches, None, flags=2, matchColor=(0, 255, 0))
    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
    is_genuine = (count >= 10) and (ratio >= 15.0)
    msg = f"⚡ 매칭률: {ratio:.1f}% ({count}점)"
    return is_genuine, count, ratio, res_img_rgb, msg

# UI
if 'artworks' not in st.session_state: st.session_state['artworks'] = [] 
if 'cars' not in st.session_state: st.session_state['cars'] = []
if 'objects' not in st.session_state: st.session_state['objects'] = []

st.set_page_config(page_title="WES Final Ver 4.3", layout="wide")
st.title("📱 WES 통합 플랫폼 [Final Ver 4.3]")
st.caption("시스템: Forensic Mode Added (정밀 모사품 대응)")

tab1, tab2, tab3 = st.tabs(["🎨 A. 진품 거래", "🚗 B. 스마트 주차", "🧸 C. 사물/미아 찾기"])

with tab1:
    st.header("🎨 미술품 진품 인증")
    c1, c2 = st.columns(2)
    with c1:
        with st.form("art_reg"):
            up = st.file_uploader("원본 등록", key="a_up")
            name = st.text_input("작품명/소유자")
            if st.form_submit_button("등록") and up:
                st.session_state['artworks'].append({"image": Image.open(up), "name": name})
                st.success(f"'{name}' 등록 완료")
    with c2:
        ver = st.file_uploader("검증", key="a_ver")
        # [New] 감별 모드 체크박스
        strict_mode = st.checkbox("🕵️ S급 모사품 감별 (초정밀 모드)", help="체크 시 오차 범위를 1.0 미만으로 줄입니다. 사진을 반듯하게 찍어야 합니다.")
        
        if ver and st.button("🔍 검증 시작"):
            t = Image.open(ver)
            st.image(t, width=200)
            with st.spinner("분석 중..."):
                bm=None; mm=0; br=0; bi=None; bmsg=""
                for art in st.session_state['artworks']:
                    # strict_mode 값 전달
                    res = match_art_forensic(art['image'], t, strict_mode)
                    if res[1] > mm: mm=res[1]; br=res[2]; bm=art; bi=res[3]; bmsg=res[4]
                if bm and res[0]: 
                    st.success(f"🎉 진품입니다! (원본: {bm['name']})")
                    st.info(bmsg); st.image(bi, use_container_width=True)
                else: 
                    st.error("🚨 가품(또는 모사품)입니다.")
                    if mm > 0: st.warning(f"유사점 {mm}개 - 구조적 불일치 ({bmsg})")

with tab2:
    st.header("🚗 스마트 주차 관제")
    c3, c4 = st.columns(2)
    with c3:
        with st.form("car_reg"):
            up = st.file_uploader("입차 차량", key="c_up")
            no = st.text_input("차량 번호")
            if st.form_submit_button("입차") and up:
                st.session_state['cars'].append({"image": Image.open(up), "no": no, "time": datetime.now()})
                st.success(f"차량 '{no}' 입차 완료")
    with c4:
        ver = st.file_uploader("출차 인식", key="c_ver")
        if ver and st.button("⚡ 정산 요청"):
            t = Image.open(ver)
            bm=None; mm=0; br=0; bi=None; bmsg=""
            for car in st.session_state['cars']:
                res = match_fast_rapid(car['image'], t)
                if res[1] > mm: mm=res[1]; br=res[2]; bm=car; bi=res[3]; bmsg=res[4]
            if bm and res[0]:
                duration = datetime.now() - bm['time']
                fee = (duration.seconds // 60 // 10) * 1000 
                st.success(f"✅ 차량 인식: {bm['no']}")
                st.info(f"주차 시간: {duration.seconds//60}분 / 요금: {fee:,}원")
                st.image(bi, use_container_width=True)
            else: st.error("🚫 인식 실패"); st.warning(bmsg)

with tab3:
    st.header("🧸 사물/미아 찾기")
    c5, c6 = st.columns(2)
    with c5:
        with st.form("obj_reg"):
            up = st.file_uploader("대상 등록", key="o_up")
            info = st.text_input("이름/연락처")
            if st.form_submit_button("등록") and up:
                st.session_state['objects'].append({"image": Image.open(up), "info": info})
                st.success(f"'{info}' 등록 완료")
    with c6:
        ver = st.file_uploader("발견물 촬영", key="o_ver")
        if ver and st.button("⚡ 보호자 찾기"):
            t = Image.open(ver)
            bm=None; mm=0; br=0; bi=None; bmsg=""
            for obj in st.session_state['objects']:
                res = match_fast_rapid(obj['image'], t)
                if res[1] > mm: mm=res[1]; br=res[2]; bm=obj; bi=res[3]; bmsg=res[4]
            if bm and res[0]:
                st.success(f"✅ 확인됨!")
                st.info(f"보호자 정보: {bm['info']}")
                st.image(bi, use_container_width=True)
            else: st.error("🚫 정보 없음"); st.warning(bmsg)
