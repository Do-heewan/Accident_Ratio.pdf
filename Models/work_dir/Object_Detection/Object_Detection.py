from mmengine import Config
from mmdet.apis import inference_detector, init_detector
import cv2
import os
import json
import argparse
import numpy as np
from glob import glob

from PIL import Image

video_name = "bb_1_020414_vehicle_187_056"

work_dir = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/work_dir/"

config_path = work_dir + "Object_Detection/config/cascade_rcnn_cfg.py"
checkpoint_path = work_dir + "Object_Detection/checkpoint/best_coco_bbox_mAP_epoch_10.pth"

input_path = work_dir + "datasets/video_data/" + video_name + "/"
output_path = work_dir + "datasets/Results/"

def detect_objects(model, image_path, save_path, score_thr=0.3):
    """이미지에서 객체를 검출하고 결과를 JSON으로 저장"""
    # 이미지 읽기
    img = cv2.imread(image_path)
    img = Image.open(image_path).convert("RGB")  # PIL로 로드
    img = np.array(img)  # OpenCV와 호환되도록 변환
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {image_path}")
        return None
    
    # 객체 검출 실행
    result = inference_detector(model, image_path)
    
    # 결과 추출
    labels = list(result.pred_instances.labels.detach().cpu().numpy())
    bboxes = list(result.pred_instances.bboxes.detach().cpu().numpy())
    scores = list(result.pred_instances.scores.detach().cpu().numpy())
    
    # 결과 저장용 리스트
    detections = []
    
    # 이미지 파일 이름 추출(경로 없이)
    image_name = os.path.basename(image_path)
    
    # 각 검출 결과에 대해
    for bbox, label, score in zip(bboxes, labels, scores):
        if score < score_thr:
            continue
            
        xmin = int(bbox[0])
        ymin = int(bbox[1])
        xmax = int(bbox[2])
        ymax = int(bbox[3])
        
        width = xmax - xmin
        height = ymax - ymin
        
        detection = {
            "image_path": image_path,
            "image_name": image_name,
            "category_id": int(label),
            "bbox": [xmin, ymin, width, height],
            "score": float(score)
        }
        
        detections.append(detection)
    
    # 결과 저장
    return detections

def main():
    # 설정 파일 로드
    cfg = config_path
    checkpoint = checkpoint_path
    device = 'cpu'
    score_thr = 0.3 # 점수 임계값
    
    # 모델 초기화
    model = init_detector(cfg, checkpoint, device=device)
    
    # 결과 저장용 리스트
    all_detections = []
    
    # 입력이 폴더인지 파일인지 확인
    if os.path.isdir(input_path):
        # 폴더 내 이미지 파일 검색 (jpg, jpeg, png)
        image_files = glob(os.path.join(input_path, "*.jpg")) + \
                     glob(os.path.join(input_path, "*.jpeg")) + \
                     glob(os.path.join(input_path, "*.png"))
        
        # 각 이미지에 대해 처리
        for image_path in image_files:
            image_path = image_path.replace("\\", "/")  # 경로 구분자 통일

            print(f"처리 중: {image_path}")
            detections = detect_objects(model, image_path, output_path, score_thr)
            if detections:
                all_detections.extend(detections)
    else:
        # 단일 이미지 파일 처리
        print(f"처리 중: {input_path}")
        detections = detect_objects(model, input_path, output_path, score_thr)
        if detections:
            all_detections.extend(detections)
    
    # 모든 검출 결과를 JSON으로 저장
    output_file = os.path.join(output_path, video_name + "_detection.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_detections, f, ensure_ascii=False, indent=4)
    
    print(f"검출 완료! 결과가 저장됨: {output_file}")
    print(f"총 {len(all_detections)}개의 객체가 검출되었습니다.")

if __name__ == "__main__":
    main()