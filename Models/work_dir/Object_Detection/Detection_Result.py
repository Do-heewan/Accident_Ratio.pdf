import cv2
import json
import os
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

video_name = "KakaoTalk_20250520_141905751"

work_dir = "c:/Users/Noh/github/Accident_Prediction_Prevent/Models/work_dir/"
data_dir = work_dir + "datasets/"

image_path = data_dir + "video_data/" + video_name + "/"
output_path = data_dir + "Results/"

pred_json_file = data_dir + "Results/" + video_name + "_detection.json"

# 출력 디렉토리 생성
output_dir = os.path.join(output_path, video_name + "_detection/")
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 목록 가져오기
image_files = sorted(glob(os.path.join(image_path, "*.png")))
print(f"이미지 파일 개수: {len(image_files)}")

# 예측 결과 로드
with open(pred_json_file, 'r') as f:
    predictions = json.load(f)

print(f"예측 결과 개수: {len(predictions)}")

# 이미지별로 예측 결과 그룹화
pred_by_image_path = {}
for pred in predictions:
    # 필요한 필드가 있는지 확인
    if 'image_path' in pred and 'category_id' in pred and 'bbox' in pred and 'score' in pred:
        img_path = pred['image_path'].replace('\\', '/')
        if img_path not in pred_by_image_path:
            pred_by_image_path[img_path] = []
        pred_by_image_path[img_path].append(pred)

print(f"그룹화된 이미지 개수: {len(pred_by_image_path)}")

# 이미지 이름별로 예측 결과 그룹화 (파일명만 사용)
pred_by_image_name = {}
for pred in predictions:
    if 'image_name' in pred and 'category_id' in pred and 'bbox' in pred and 'score' in pred:
        img_name = pred['image_name']
        if img_name not in pred_by_image_name:
            pred_by_image_name[img_name] = []
        pred_by_image_name[img_name].append(pred)

print(f"이미지 이름별 그룹화된 개수: {len(pred_by_image_name)}")

# 시각화 함수
def visualize_predictions(image_path, annotations, output_path):
    """이미지에 바운딩 박스를 그리고 결과를 저장합니다"""
    try:
        # PIL로 이미지 로드 (한글 경로 지원)
        img = Image.open(image_path).convert("RGB")
        
        # NumPy 배열로 변환
        img = np.array(img)
        
        # 색상 정의
        colors = {
            0: (128, 0, 128),   # 보라색 - two-wheeled-vehicle
            1: (255, 0, 0),     # 빨간색 - traffic-light-red
            2: (0, 255, 0),     # 초록색 - pedestrian
            3: (255, 255, 0),   # 노란색 - crosswalk
            4: (0, 255, 255),   # 하늘색 - bike
            5: (0, 128, 0),     # 짙은 초록 - traffic-light-green
            6: (0, 0, 255),     # 파란색 - vehicle
            7: (255, 165, 0),   # 주황색 - traffic-light-etc
            8: (255, 255, 255)  # 흰색 - traffic-sign
        }
        
        # 바운딩 박스 그리기
        for ann in annotations:
            bbox = ann["bbox"]
            # bbox가 올바른 형식인지 확인
            if isinstance(bbox, list) and len(bbox) == 4:
                cat_id = ann["category_id"]
                color = colors.get(cat_id, (200, 200, 200))  # 기본 색상은 회색
                
                # 좌표 추출 및 정수로 변환
                x, y, w, h = [int(coord) for coord in bbox]
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # 카테고리와 점수 텍스트 추가
                text = f"cat:{cat_id} {ann['score']:.2f}"
                cv2.putText(img, text, (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 결과 저장 - 디렉토리 확인 및 생성
        output_dir = os.path.dirname(output_path)
        if output_dir:  # 디렉토리가 비어있지 않은 경우에만
            os.makedirs(output_dir, exist_ok=True)
        
        # RGB에서 BGR로 변환 (OpenCV 형식)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 직접 경로 처리 후 저장
        clean_output_path = output_path.replace('\\', '/')
        print(f"저장 시도: {clean_output_path}")
        
        # 이미지 저장 (파일명을 명시적으로 처리)
        output_filename = os.path.basename(clean_output_path)
        output_dirname = os.path.dirname(clean_output_path)
        if not output_dirname:  # 디렉토리가 지정되지 않은 경우
            output_dirname = '.'  # 현재 디렉토리
            
        final_path = os.path.join(output_dirname, output_filename)
        
        # 저장 시도 - 여러 방법으로
        try:
            # 방법 1: OpenCV로 저장
            success = cv2.imwrite(final_path, img_bgr)
            if not success:
                # 방법 2: PIL로 저장
                Image.fromarray(img).save(final_path)
                print(f"PIL로 저장 성공: {final_path}")
            else:
                print(f"OpenCV로 저장 성공: {final_path}")
            return True
        except Exception as save_err:
            print(f"저장 오류: {save_err}")
            return True
            
    except Exception as e:
        print(f"이미지 {image_path} 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()  # 상세한 오류 정보 출력
        return False

# 이미지 파일 목록을 하나씩 처리
print("\n이미지 파일 목록으로 시각화 중...")
successful_count = 0

for img_file in tqdm(image_files):
    img_name = os.path.basename(img_file)
    img_file_normalized = img_file.replace('\\', '/')

    if img_name in pred_by_image_name:
        # 이미지 이름으로 매치
        annotations = pred_by_image_name[img_name]
        output_path = os.path.join(output_dir, f"pred_{img_name}")
        if visualize_predictions(img_file, annotations, output_path):
            successful_count += 1
    else:
        print(f"이미지 {img_name}에 대한 예측 결과를 찾을 수 없습니다.")

print(f"\n성공적으로 {successful_count}개 이미지를 처리했습니다.")
print(f"결과가 {output_dir}에 저장되었습니다.")