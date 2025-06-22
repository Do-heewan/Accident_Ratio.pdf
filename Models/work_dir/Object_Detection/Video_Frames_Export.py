import cv2
import os
from tqdm import tqdm
from PIL import Image  # PIL Image 모듈 추가

video_name = "" # video name
video_path = "" # work directory path

def extract_frames(video_path, output_dir, frame_interval=1):
    """비디오에서 프레임 추출"""
    try:
        # 비디오 캡처 객체 생성 및 검증
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 비디오 정보 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"비디오 크기: {width}x{height}")
        print(f"총 프레임 수: {total_frames}")

        frame_count = 1
        saved_count = 1

        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # 프레임 이름 생성 (saved_count가 아닌 frame_count 사용)
                    frame_name = f"{frame_count:03d}.png"  # jpg 대신 png 사용
                    frame_path = os.path.join(output_dir, frame_name)
                    
                    try:
                        # 이미지 저장 전 검증
                        if frame is None or frame.size == 0:
                            print(f"\n유효하지 않은 프레임 데이터: {frame_count}")
                            continue
                            
                        # BGR에서 RGB로 변환
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # PIL Image로 변환하여 저장
                        image = Image.fromarray(frame_rgb)
                        image.save(frame_path)
                        
                        saved_count += 1
                        if saved_count == 1:
                            print(f"\n첫 프레임 저장 성공: {frame_path}")
                            print(f"프레임 크기: {frame.shape}")
                            
                    except Exception as e:
                        print(f"\n프레임 {frame_count} 저장 중 오류: {str(e)}")

                frame_count += 1
                pbar.update(1)

        cap.release()
        
        # 저장 결과 확인
        saved_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        print(f"\n총 처리된 프레임: {frame_count}")
        print(f"저장된 프레임: {len(saved_files)}")
        
        return saved_count

    except Exception as e:
        print(f"처리 중 오류 발생: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return 0
    
# 메인 실행 코드
if __name__ == "__main__":    
    video_path = os.path.join(video_path, video_name)
    
    # 출력 디렉토리 설정 (비디오 파일명 기반)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), f"{base_name}")
    
    try:
        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일이 없습니다: {video_path}")
            
        print(f"입력 비디오: {video_path}")
        print(f"출력 경로: {output_dir}")
        
        saved_frames = extract_frames(video_path, output_dir)
        
        if saved_frames > 0:
            print(f"\n성공: {saved_frames}개 프레임 저장")
        else:
            print("\n실패: 프레임을 저장하지 못했습니다")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")